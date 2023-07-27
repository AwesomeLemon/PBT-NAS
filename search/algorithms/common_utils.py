import glob
import time

import numpy as np
import os
import pickle
import random
import shutil
from copy import deepcopy
from pathlib import Path

import ray
import yaml

import utils
import utils.rsync_wrapper

import utils.soup

import utils.general
from data_providers import get_data_provider
from models import make_function_create_model
from utils.general import rmtree_if_exists
from search.gan.fitness_fun_gan import GanFitness
from search.hyperparameters import Hyperparameters
from search.rl.fitness_fun_rl import RlFitness
from search_space import get_arch_search_space


def init_run(args):
    utils.general.set_random_seeds(args.seed)
    arch_ss = get_arch_search_space(**vars(args))
    hyperparameters = Hyperparameters(arch_ss, **vars(args))
    alphabet_array = np.array(hyperparameters.alphabet)
    basic_folder = args.folder
    print(f'{basic_folder=}')
    N_MODELS = int(args.population_size)

    continue_epoch = None if not hasattr(args, 'continue_epoch') else args.continue_epoch
    continue_auto = None if not hasattr(args, 'continue_auto') else args.continue_auto

    if continue_auto is not None and continue_epoch is None:
        if continue_auto:
            # find the last population file
            found_epochs = [int(f_name[f_name.rfind('_') + 1:f_name.rfind('.pkl')]) for f_name in
                            glob.glob(os.path.join(basic_folder, "population_*.pkl"))]
            if len(found_epochs) > 0:
                continue_epoch = max(found_epochs)

    if_shared_fs = False if not hasattr(args, 'if_shared_fs') else args.if_shared_fs
    if continue_epoch is None:
        if_not_clean_dir_bohb = False if not hasattr(args, 'if_not_clean_dir_bohb') else args.if_not_clean_dir_bohb
        fn_for_dir = create_or_clean_dir if not if_not_clean_dir_bohb else create_dir
        if if_shared_fs:
            fn_for_dir(os.path.join(basic_folder, 'models'))
            fn_for_dir(os.path.join(basic_folder, 'tensorboard'))
        else:
            utils.general.ray_run_fun_once_per_node(fn_for_dir, os.path.join(basic_folder, 'models'))
            utils.general.ray_run_fun_once_per_node(fn_for_dir, os.path.join(basic_folder, 'tensorboard'))
        population = [{'model_id': model_id, 'fitnesses':[]} for model_id in range(N_MODELS)]
        population = randomly_init_population_hparams(population, alphabet_array, args)
    else:
        population = pickle.load(open(os.path.join(basic_folder, f'population_{continue_epoch}.pkl'), 'rb'))

    print(population)
    data_provider = get_data_provider(**vars(args))
    kwargs_for_function_create_model = deepcopy(vars(args))
    kwargs_for_function_create_model['model_parameters']['data_provider'] = data_provider
    kwargs_for_function_create_model['model_parameters']['arch_ss'] = arch_ss
    function_create_model = make_function_create_model(**kwargs_for_function_create_model)

    final_upload_node = None if not hasattr(args, 'final_upload_node') else args.final_upload_node
    if if_shared_fs and final_upload_node is None:
        ssh_user, ray_head_node = None, None # don't need the values
    else:
        ssh_user, ray_head_node = args.ssh_user, args.ray_head_node

    rsync_wrapper = utils.rsync_wrapper.RsyncWrapper(ssh_user, ray_head_node, if_shared_fs, final_upload_node)

    MAX_EPOCHS, EPOCHS_STEP = int(args.max_epochs), int(args.epochs_step)
    task = args.model_parameters['task']
    shrink_perturb_always = None if not hasattr(args, 'shrink_perturb_always') else args.shrink_perturb_always
    if_gen_avg = False if not hasattr(args, 'if_gen_avg') else args.if_gen_avg
    evaluate_mode = False if not hasattr(args, 'evaluate_mode') else args.evaluate_mode
    num_cpus = 4 if not hasattr(args, 'num_cpus') else args.num_cpus
    emulate_distributed = False if not hasattr(args, 'emulate_distributed') else args.emulate_distributed
    n_machines_to_emulate = None if not hasattr(args, 'n_machines_to_emulate') else args.n_machines_to_emulate
    cleanup_final = True if not hasattr(args, 'cleanup_final') else args.cleanup_final
    rl_unshared_buffer = task != 'generation' and args.rl_general['replay_buffer_type'] == 'ram'

    st = time.time()
    if task == 'generation':
        fitness_function = GanFitness(basic_folder, MAX_EPOCHS, EPOCHS_STEP, data_provider, hyperparameters,
                                      function_create_model,
                                      args, shrink_perturb_always, if_gen_avg)
    else:
        fitness_function = RlFitness(basic_folder, MAX_EPOCHS, EPOCHS_STEP, hyperparameters,
                                     function_create_model, args, shrink_perturb_always)
    print(f'fitness_function init time {time.time() - st:.2f}')

    return N_MODELS, alphabet_array, basic_folder, population, data_provider, function_create_model, \
           hyperparameters, continue_epoch, arch_ss, rsync_wrapper, if_shared_fs, MAX_EPOCHS, EPOCHS_STEP, \
           task, shrink_perturb_always, if_gen_avg, evaluate_mode, num_cpus, emulate_distributed, \
           n_machines_to_emulate, cleanup_final, rl_unshared_buffer, fitness_function


def randomly_init_population_hparams(population, alphabet_array, args):
    # fixed_hparams is useful for debugging: set values to the ones you want.
    fixed_hparams = None if not hasattr(args, 'fixed_hparams') else args.fixed_hparams
    fixed_hparams_start = [0] if not hasattr(args, 'fixed_hparams_start') else args.fixed_hparams_start
    init_genepool = []
    for k in range(len(alphabet_array)):
        n_per_option = len(population) // alphabet_array[k]
        cur_pool = []
        for option in range(alphabet_array[k]):
            cur_pool += [option] * n_per_option
        while len(cur_pool) < len(population): # if population size is not divisible by the number of options, pad
            cur_pool.append(np.random.randint(alphabet_array[k]))
        random.shuffle(cur_pool)
        init_genepool.append(cur_pool)
    for i in range(len(population)):
        hparams = []
        for k in range(len(alphabet_array)):
            hparams.append(init_genepool[k][i])

        if fixed_hparams is not None:
            assert len(fixed_hparams_start) == len(fixed_hparams)
            for start_ind, fixed_hparams_part in zip(fixed_hparams_start, fixed_hparams):
                for j in range(len(fixed_hparams_part)):
                    hparams[start_ind + j] = int(fixed_hparams_part[j])

        population[i]['hyperparameters'] = np.copy(hparams)
    return population


def delete_old_model_files(basic_folder, epoch, delete_backups_only=False):
    model_dir_path = '%s/models' % basic_folder
    if os.path.exists(model_dir_path):
        model_files = os.listdir('%s/models' % basic_folder)
        for file_name in model_files:
            if '_backup' in file_name:
                print(f'delete {file_name}')
                os.remove('%s/models/%s' % (basic_folder, file_name))
            elif not delete_backups_only:
                if 'workdir' in file_name:  # don't delete RL workdirs
                    continue
                file_epoch = int(file_name.split('_')[2])
                if file_epoch != epoch:
                    print(f'delete {file_name}')
                    os.remove('%s/models/%s' % (basic_folder, file_name))
    else:
        # niche use case: a run was started on a single node, but I wanna continue it on many nodes.
        # Then 'models' dir doesn't exist on them
        Path(model_dir_path).mkdir()


def create_or_clean_dir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    Path(folder).mkdir()

def create_dir(folder):
    if not os.path.exists(folder):
        Path(folder).mkdir()

def find_best(population):
    best_fitness = -1e10
    best_index = None
    for i_p, p in enumerate(population):
        if p['fitnesses'][-1] > best_fitness:
            best_fitness = p['fitnesses'][-1]
            best_index = p['model_id']
            assert i_p == best_index, 'just to be sure'
    return best_fitness, best_index


def evaluate_store_clean(MAX_EPOCHS, args, basic_folder, cleanup_final, fitness_function_ref, if_shared_fs, population,
                         rsync_wrapper, soups, task):
    # copy into the main folder the images generated by the best model / video from the best agent
    best_fitness, best_model_id = find_best(population)

    if task == 'generation':
        if not os.path.exists(os.path.join(basic_folder, f'{best_model_id}.png')):  # if the image hasn't already been
            #                                                               copied, and the 'generated' dir deleted
            copy_from = os.path.join(basic_folder, 'generated', str(best_model_id), f'{MAX_EPOCHS}.png')
            shutil.copy(copy_from, os.path.join(basic_folder, f'{best_model_id}.png'))
    elif task == 'rl_normact':
        video_dir = os.path.join(basic_folder, 'models', f'workdir_{best_model_id}', 'eval_video')
        video_files = glob.glob(video_dir + '/*')
        latest_file = max(video_files, key=os.path.getctime)
        shutil.copy(latest_file, os.path.join(basic_folder, f'{best_model_id}.mp4'))

    for soup_type, soup_target in soups:
        utils.soup.cook_model_soup_greedy_ray(basic_folder, fitness_function_ref, args, soup_type, soup_target, 'last')

    if task == 'generation':
        for soup_type, soup_target in soups:
            utils.soup.cook_model_soup_greedy_ray(basic_folder, fitness_function_ref, args, soup_type, soup_target, 'ever')

        utils.general.evaluate_final_gan_ray(basic_folder, fitness_function_ref, args, soups)
        if cleanup_final:
            if if_shared_fs:
                rmtree_if_exists(os.path.join(basic_folder, 'generated'))
            else:
                utils.general.ray_run_fun_once_per_node(rmtree_if_exists, os.path.join(basic_folder, 'generated'))

    if cleanup_final:
        if if_shared_fs:
            rmtree_if_exists(os.path.join(basic_folder, 'models'))
            rmtree_if_exists(os.path.join(basic_folder, 'models_best_iter'))
        else:
            utils.general.ray_run_fun_once_per_node(rmtree_if_exists, os.path.join(basic_folder, 'models'))
            utils.general.ray_run_fun_once_per_node(rmtree_if_exists, os.path.join(basic_folder, 'models_best_iter'))

    rsync_wrapper.upload_final(basic_folder, True)


def store_population_and_best_weights(basic_folder, best_fitness_ever, next_epoch, population, soups, task):
    # population
    with open(os.path.join(basic_folder, f'population_{next_epoch}.pkl'), 'wb') as f:
        pickle.dump(population, f)
    population_for_yaml = deepcopy(population)
    for p in population_for_yaml:
        p['hyperparameters'] = p['hyperparameters'].tolist()
    yaml.safe_dump(population_for_yaml, open(os.path.join(basic_folder, f'population_{next_epoch}.yml'), 'w'),
                   default_flow_style=None)

    # best weights
    best_fitness_cur, best_index_cur = find_best(population_for_yaml)
    info_best = {'epoch': next_epoch, 'pop_member': population_for_yaml[best_index_cur]}
    yaml.safe_dump(info_best, open(os.path.join(basic_folder, 'best_last_info.yml'), 'w'), default_flow_style=None)
    copy_from = os.path.join(basic_folder, 'models', f'model_{best_index_cur}_{next_epoch}')
    shutil.copy(copy_from, os.path.join(basic_folder, 'best_last_model'))
    if best_fitness_cur > best_fitness_ever:
        best_fitness_ever = best_fitness_cur
        yaml.safe_dump(info_best, open(os.path.join(basic_folder, 'best_ever_info.yml'), 'w'), default_flow_style=None)
        copy_from = os.path.join(basic_folder, 'models', f'model_{best_index_cur}_{next_epoch}')
        shutil.copy(copy_from, os.path.join(basic_folder, 'best_ever_model'))

        if len(soups) > 0 and task == 'generation':
            models_best_iter_dir = os.path.join(basic_folder, 'models_best_iter')
            if os.path.exists(models_best_iter_dir):
                shutil.rmtree(models_best_iter_dir)
            Path(models_best_iter_dir).mkdir()
            for file_name in os.listdir(os.path.join(basic_folder, 'models')):
                if 'model' in file_name:
                    file_epoch = int(file_name.split('_')[2])
                    if file_epoch == next_epoch:
                        shutil.copy(os.path.join(basic_folder, 'models', file_name), models_best_iter_dir)
    return best_fitness_ever


@ray.remote(num_cpus=4, num_gpus=1, max_calls=1)
def update_one_model(i, args, basic_folder, epoch, fitness_function, next_epoch, cur_pop_member, task,
                     rsync_wrapper_ref, emulation_i=None):
    import warnings
    warnings.filterwarnings("ignore", message="using `dtype=` in comparisons is only useful for `dtype=object`")

    cur_seed = args.seed * 100000 + epoch * 1000 + i * 10
    utils.general.set_random_seeds(cur_seed)

    cur_pop_member = deepcopy(cur_pop_member)
    model_id = cur_pop_member['model_id']
    print('current model', model_id)

    ckpt_full_path = os.path.join(basic_folder, 'models', f'model_{i}_{epoch}')
    if epoch > 0: # on the 0-th epoch there's no checkpoint yet.
        # there are some time shenanigans going on with rsync, better delete old ckpt manually
        if not (rsync_wrapper_ref.if_shared_fs or rsync_wrapper_ref.ray_head_node == rsync_wrapper_ref._get_name_this_node()):
            if os.path.exists(ckpt_full_path):
                os.remove(ckpt_full_path)
        rsync_wrapper_ref.download(ckpt_full_path)
        print(f'Downloaded {ckpt_full_path}')
        if task != 'generation':
            # need to download it in case the individual was replaced with a child
            # (so a buffer was copied from the parent:
            #   - in the unshared case, copying the buffer is critical
            #   - in the shared case, workdir_i contains no buffer, but still contains eval videos)
            rsync_wrapper_ref.download(os.path.join(basic_folder, 'models', f'workdir_{i}'), if_dir=True)

    fitness_args = (cur_pop_member['hyperparameters'], model_id, epoch)
    fitness_kwargs = dict(save_at_epoch_0=True, seed=cur_seed, rsync_wrapper_ref=rsync_wrapper_ref)
    if emulation_i is not None:
        fitness_kwargs['emulation_i'] = emulation_i

    fitness_value = fitness_function.fitness(*fitness_args, **fitness_kwargs)
    print(f'computed fitness for model {model_id}')
    cur_pop_member['fitnesses'].append(fitness_value)

    ckpt_full_path_next_epoch = os.path.join(basic_folder, 'models', f'model_{i}_{next_epoch}')
    rsync_wrapper_ref.upload(ckpt_full_path_next_epoch)

    if task == 'generation':
        path_to_generated_dir = os.path.join(basic_folder, 'generated', str(model_id))
        shutil.move(os.path.join(path_to_generated_dir, f'{next_epoch}_tmp.png'),
                    os.path.join(path_to_generated_dir, f'{next_epoch}.png'))

        # remove _tmp images
        files = os.listdir(path_to_generated_dir)
        for file_name in files:
            if '_tmp.png' in file_name:
                os.remove(os.path.join(path_to_generated_dir, file_name))

        rsync_wrapper_ref.upload(path_to_generated_dir, True)

    return cur_pop_member


def train_population(individuals, i_offset, args, basic_folder, epoch,
                     fitness_function, next_epoch, task, rsync_wrapper_ref, num_gpus, num_cpus, emulate_distributed,
                     n_machines_to_emulate, rl_unshared_buffer):
    print(ray.nodes())
    results = []

    futures = []
    try:
        if not emulate_distributed:
            for i in np.random.permutation(range(len(individuals))):
                ray_options = dict(num_gpus=num_gpus, num_cpus=num_cpus)
                runtime_env = {"env_vars": {"EGL_DEVICE_ID": "1"}}
                if rl_unshared_buffer: # force the model to always be trained on the same machine
                    ray_options['resources'] = {f'{i}': 1}
                futures.append(
                    update_one_model.options(**ray_options, runtime_env=runtime_env)
                                    .remote(i + i_offset, args, basic_folder, epoch, fitness_function,
                                            next_epoch, individuals[i], task, rsync_wrapper_ref))


            for f in futures:
                new_population_member = ray.get(f) # wait on everything
                results.append(new_population_member)

        else:
            assert len(individuals) % n_machines_to_emulate == 0
            assert not rl_unshared_buffer

            n_per_machine = len(individuals) // n_machines_to_emulate
            cnt = 0

            for i in np.random.permutation(range(len(individuals))):
                futures.append(
                    update_one_model.options(num_gpus=num_gpus, num_cpus=num_cpus)#, runtime_env=runtime_env)
                                    .remote(i + i_offset, args, basic_folder, epoch, fitness_function, next_epoch,
                                            individuals[i], task, rsync_wrapper_ref, emulation_i=cnt // n_per_machine))

                # assuming a single node, since that's the point of emulation
                cnt += 1
                if cnt % n_per_machine == 0:
                    for f in futures:
                        new_population_member = ray.get(f)
                        results.append(new_population_member)
                    futures = []

            # just in case
            for f in futures:
                new_population_member = ray.get(f)  # wait on everything
                results.append(new_population_member)

    except Exception as e:
        print('Exception in one of the tasks')
        for f in futures:
            ray.cancel(f, force=True)
        raise e


    results = sorted(results, key=lambda p: p['model_id'])
    return results


def init_best_fitness_ever(basic_folder, continue_epoch):
    best_fitness_ever = -1e10
    if continue_epoch is not None:
        best_ever_dict_path = os.path.join(basic_folder, 'best_ever_info.yml')
        if os.path.exists(best_ever_dict_path):
            best_ever_dict = yaml.safe_load(open(best_ever_dict_path))
            best_fitness_ever = best_ever_dict['pop_member']['fitnesses'][-1]
    return best_fitness_ever
