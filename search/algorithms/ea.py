import copy
import numpy as np
import os
import ray
import shutil
import time
import yaml
from copy import deepcopy
from functools import partial
from shutil import copyfile

import utils
import utils.general
from search.algorithms.common_utils import init_run, delete_old_model_files, evaluate_store_clean, \
    store_population_and_best_weights, train_population, init_best_fitness_ever
from search.algorithms.mix import copy_weights, copy_weights_mutate, genes_to_layer_names_gan, genes_to_layer_names_rl
from search.algorithms.mutation import mutate_weights
from search_space import RlNormActSearchSpace
from utils.general import print_population, rmtree_if_exists


def run_EA(args):
    st = time.time()
    n_parents, alphabet_array, basic_folder, population, data_provider, \
        function_create_model, hyperparameters, continue_epoch, arch_ss, rsync_wrapper, if_shared_fs, \
        MAX_EPOCHS, EPOCHS_STEP, task, shrink_perturb_always, if_gen_avg, evaluate_mode, num_cpus, \
        emulate_distributed, n_machines_to_emulate, cleanup_final, rl_unshared_buffer, fitness_function = init_run(args)
    print(f'Init time {time.time() - st:.2f}')

    n_children = args.n_children
    replace_arch_percent = 0.25 if not hasattr(args, 'replace_arch_percent') else args.replace_arch_percent
    percent_old_population_survive = args.percent_old_population_survive
    percent_parents_for_mixing = args.percent_parents_for_mixing
    if_mutate = False if not hasattr(args, 'if_mutate') else args.if_mutate
    mix_with_itself = False if not hasattr(args, 'mix_with_itself') else args.mix_with_itself

    fn_if_mix = (lambda _: True) if not hasattr(args, 'fn_if_mix') else eval(args.fn_if_mix) # next_epoch -> True/False
    shrink_perturb = None if not hasattr(args, 'shrink_perturb') else args.shrink_perturb
    soups = [] if not hasattr(args, 'soups') else args.soups

    keep_old_checkpoints = False if not hasattr(args, 'keep_old_checkpoints') else args.keep_old_checkpoints

    rsync_wrapper_ref = ray.put(rsync_wrapper)

    fun_create_perturb_source_from_hps = partial(fun_create_perturb_source_from_hps_random_init, fitness_function,
                                                 type(arch_ss) is RlNormActSearchSpace)

    global_history = init_global_history(EPOCHS_STEP, arch_ss, basic_folder, continue_epoch, population)

    processes_per_gpu = args.processes_per_gpu
    num_gpus = 1 / processes_per_gpu

    if evaluate_mode:
        utils.general.evaluate_final_gan_ray(basic_folder, fitness_function, args, [])
        return

    best_fitness_ever = init_best_fitness_ever(basic_folder, continue_epoch)

    epoch = 0 if continue_epoch is None else continue_epoch
    while epoch < MAX_EPOCHS:
        if if_shared_fs:
            delete_old_model_files(basic_folder, epoch, keep_old_checkpoints)
        else:
            utils.general.ray_run_fun_once_per_node(delete_old_model_files, basic_folder, epoch, keep_old_checkpoints)

        next_epoch = epoch + EPOCHS_STEP

        parents_evaluated = train_population(population, 0, args, basic_folder, epoch, fitness_function, next_epoch,
                                             task, rsync_wrapper_ref, num_gpus, num_cpus, emulate_distributed,
                                             n_machines_to_emulate, rl_unshared_buffer)
        print_population(parents_evaluated, 'parents after training')

        if (not fn_if_mix(next_epoch)) or (epoch == MAX_EPOCHS - EPOCHS_STEP):
            new_population = parents_evaluated
        else:
            # create children
            children, info_per_child = create_children(arch_ss, basic_folder, epoch, EPOCHS_STEP, n_children,
                                                       n_parents, parents_evaluated, population, replace_arch_percent,
                                                       shrink_perturb, percent_parents_for_mixing, mix_with_itself,
                                                       fun_create_perturb_source_from_hps, if_mutate, alphabet_array,
                                                       args.seed)
            if not if_mutate:
                # need to set some values
                for child in children:
                    child['fitnesses'].append(-1e3)

            # selection
            indices_selected = select(children, n_parents, parents_evaluated, percent_old_population_survive)

            # renumber children & rename their checkpoints
            new_population, child_old_id_to_new_id = update_population_and_checkpoints(basic_folder, info_per_child,
                                                               children, indices_selected, n_parents, next_epoch,
                                                               parents_evaluated, arch_ss)
            # Important: update_population_and_checkpoints breaks order, after it index in the population != model_id
            new_population = sorted(new_population, key=lambda individual: individual['model_id'])

            # update global_history
            update_global_history(basic_folder, child_old_id_to_new_id, children, global_history, indices_selected,
                                  info_per_child, n_parents, next_epoch, parents_evaluated)

        print_population(new_population, 'New population')
        population = deepcopy(new_population)

        best_fitness_ever = store_population_and_best_weights(basic_folder, best_fitness_ever, next_epoch,
                                                              population, soups, task)

        epoch += EPOCHS_STEP
        print(f'{basic_folder}')

    evaluate_store_clean(MAX_EPOCHS, args, basic_folder, cleanup_final, fitness_function, if_shared_fs, population,
                         rsync_wrapper, soups, task)


def init_global_history(EPOCHS_STEP, arch_ss, basic_folder, continue_epoch, population):
    def _create_from_scratch(arch_ss, population):
        global_history = []
        for p in population:  # first entry in each block's history is its original model
            p['history'] = {i: [p['model_id']] for i in range(len(arch_ss.get_n_options_per_gene()))}
            global_history.append(dict(seq_id=len(global_history), model_id=p['model_id'], epoch=0,
                                       parent1=None, parent2=None, indices=[],
                                       old_values=[], new_values=[],
                                       epoch_final=None, fitness_final=None))
        return global_history

    if continue_epoch is None:
        global_history = _create_from_scratch(arch_ss, population)
    else:
        epoch_for_history = continue_epoch
        while (
        not os.path.exists(os.path.join(basic_folder, f'history_{epoch_for_history}.yml'))) and epoch_for_history > 0:
            epoch_for_history -= EPOCHS_STEP
        if epoch_for_history > 0:
            global_history = yaml.safe_load(open(os.path.join(basic_folder, f'history_{epoch_for_history}.yml')))
        else:
            global_history = _create_from_scratch(arch_ss, population)
    return global_history

def create_children(arch_ss, basic_folder, epoch, epoch_step, n_children, n_parents, parents_evaluated,
                    population, replace_arch_percent, shrink_perturb, percent_parents_for_mixing, mix_with_itself,
                    fun_create_perturb_source_from_hps,
                    if_mutate, alphabet_array, seed):
    children_not_evaluated = []
    info_per_child = {}

    n_iterations = n_children # 1 iter = 1 child

    assert replace_arch_percent >= 0
    assert replace_arch_percent <= 1

    fitnesses_parents = [p['fitnesses'][-1] for p in parents_evaluated]
    indices_sorted = np.argsort(fitnesses_parents)
    indices_top = indices_sorted[-int(n_parents * percent_parents_for_mixing):].tolist()

    child_weights_futures = []
    for i in range(n_iterations):
        child_info = {}
        # select parents
        parent1_i, parent2_i = np.random.choice(indices_top, 2, replace=False)
        if mix_with_itself:
            parent2_i = parent1_i
        parent1_i, parent2_i = int(parent1_i), int(parent2_i) # from np.int64 to int - just to be safe
        parent1_fitness, parent2_fitness = fitnesses_parents[parent1_i], fitnesses_parents[parent2_i]

        if parent2_fitness > parent1_fitness:
            parent1_i, parent2_i = parent2_i, parent1_i
            parent1_fitness, parent2_fitness = parent2_fitness, parent1_fitness

        old_values, new_values = [], []
        if not if_mutate: # PBT-NAS mixing
            child = deepcopy(population[parent1_i])  # note that I copy the not-evaluated parent
            #                                          because I don't need the last fitness value in the child

            hps_donor = parents_evaluated[parent2_i]['hyperparameters']
            all_changed_indices = [ind_hp for ind_hp in range(len(hps_donor))
                                   if np.random.uniform() < replace_arch_percent]

            architecture_indices_to_copy = []
            for ind_hp in all_changed_indices:
                old_values.append(int(child['hyperparameters'][ind_hp]))
                new_values.append(int(hps_donor[ind_hp]))
                architecture_indices_to_copy.append(ind_hp)

                child['hyperparameters'][ind_hp] = hps_donor[ind_hp]
                child['history'][ind_hp] = copy.copy(parents_evaluated[parent2_i]['history'][ind_hp])

            # Need to change the id for creating model checkpoints
            # Will re-number after selection (& rename model checkpoints)
            child['model_id'] = n_parents + i
            print(f"Child {child['model_id']}: Mixing {parent1_i} with {parent2_i}")

            child_info['ind_changed'] = architecture_indices_to_copy
            child_info['parent1'] = parent1_i
            child_info['parent2'] = parent2_i

            f = create_child_weights.remote(arch_ss, architecture_indices_to_copy, basic_folder, child, epoch + epoch_step,
                                 parent1_i, parent2_i, shrink_perturb,
                                 fun_create_perturb_source_from_hps, False, seed)
            child_weights_futures.append(f)

            children_not_evaluated.append(child)

        else: # SEARL-like mutation
            print(f'Mutate {parent1_i}')
            child = deepcopy(parents_evaluated[parent1_i])  # note that I copy the evaluated parent
            #                                                 because I need the last fitness value in the child

            prob_each = 1 / 3
            if_mutate_weights = False
            rnd_mut_type = np.random.uniform()
            if rnd_mut_type < prob_each:
                all_changed_indices = []
            elif rnd_mut_type < prob_each * 2:
                all_changed_indices = []
                if_mutate_weights = True
            else:
                all_changed_indices = [np.random.choice(len(child['hyperparameters']))]

            architecture_indices_to_copy = []
            for ind_hp in all_changed_indices:
                new_value = np.random.choice(alphabet_array[ind_hp])
                while new_value == child['hyperparameters'][ind_hp]:
                    new_value = np.random.choice(alphabet_array[ind_hp])

                old_values.append(int(child['hyperparameters'][ind_hp]))
                new_values.append(int(new_value))

                child['hyperparameters'][ind_hp] = new_value
                architecture_indices_to_copy.append(ind_hp)

            # Need to change the id for creating model checkpoints
            # Will re-number after selection (& rename model checkpoints)
            child['model_id'] = n_parents + i

            child_info['ind_changed'] = architecture_indices_to_copy
            child_info['parent1'] = parent1_i
            child_info['parent2'] = None

            # note that I pass parent1_i twice. The second parent won't be used,
            # instead random values will be gotten from fun_create_perturb_source_from_hps,
            f = create_child_weights.remote(arch_ss, architecture_indices_to_copy, basic_folder, child,
                                 epoch + epoch_step,
                                 parent1_i, parent1_i, shrink_perturb,
                                 fun_create_perturb_source_from_hps, True, seed, if_mutate_weights)
            child_weights_futures.append(f)

            children_not_evaluated.append(child)

        child_info['old_values'] = old_values
        child_info['new_values'] = new_values

        info_per_child[child['model_id']] = child_info

    for f in child_weights_futures:
        ray.get(f)

    return children_not_evaluated, info_per_child

@ray.remote(num_cpus=4, max_calls=1, resources={'head_node':1})
def create_child_weights(arch_ss, architecture_indices_to_copy, basic_folder, child, epoch, parent_main, parent_extra,
                         shrink_perturb, fun_create_perturb_source_from_hps,
                         if_mutate, seed, if_mutate_weights=False):
    cur_seed = seed * 100000 + epoch * 1000 + parent_main * 100 + parent_extra
    utils.general.set_random_seeds(cur_seed)
    # First, copy weights of parent1 - they will be the base
    src_file = os.path.join(basic_folder, 'models', f'model_{parent_main}_{epoch}')
    dst_file = os.path.join(basic_folder, 'models', f'model_{child["model_id"]}_{epoch}')
    print(f'Copy {parent_main} to child {child["model_id"]}: copy', src_file, dst_file)
    copyfile(src_file, dst_file)
    print(f'{architecture_indices_to_copy=}')

    rl = False
    fn_genes_to_layer_names = genes_to_layer_names_gan
    if type(arch_ss) is RlNormActSearchSpace:
        fn_genes_to_layer_names = genes_to_layer_names_rl
        rl = True

    def copy_workdir():
        src_workdir_full_path = os.path.join(basic_folder, 'models', f'workdir_{parent_extra}')
        dst_workdir_full_path = os.path.join(basic_folder, 'models', f'workdir_{child["model_id"]}')

        print(f'{src_workdir_full_path=} {dst_workdir_full_path=}')
        assert os.path.exists(src_workdir_full_path)
        if os.path.exists(dst_workdir_full_path):
            print('deleting dst workdir')
            shutil.rmtree(dst_workdir_full_path)

        shutil.copytree(src_workdir_full_path, dst_workdir_full_path)

    if if_mutate and len(architecture_indices_to_copy) == 0:
        if if_mutate_weights:
            # mutate only weights using the same algorithm as SEARL
            mutate_weights(dst_file)
        # workdir
        if rl:
            copy_workdir()

        print(f"Created weights of {child['model_id']}")
        return

    module_dict_keys_to_copy = fn_genes_to_layer_names(architecture_indices_to_copy, arch_ss)
    print(f'{module_dict_keys_to_copy=}')

    model_for_perturb_state_dict = fun_create_perturb_source_from_hps(child['hyperparameters']).state_dict()

    if rl:
        copy_workdir()

    if not if_mutate:
        copy_weights(parent_extra, child['model_id'], basic_folder, epoch, module_dict_keys_to_copy,
                     shrink_perturb, model_for_perturb_state_dict, rl)
    else:
        copy_weights_mutate(parent_extra, child['model_id'], basic_folder, epoch,
                            module_dict_keys_to_copy, model_for_perturb_state_dict, rl)

    print(f"Created weights of {child['model_id']}")


def fun_create_perturb_source_from_hps_random_init(fitness_function, is_rl, hps):
    fitness_function.hyperparameters.convert_encoding_to_hyperparameters(hps)
    if not is_rl:
        model_for_perturb_state_dict = fitness_function.function_create_model(hyperparameters=fitness_function.hyperparameters)
    else:
        model_for_perturb_state_dict = fitness_function.function_create_model(hyperparameters=fitness_function.hyperparameters,
                                                                              obs_shape=(9, 84, 84), #hardcoded obs_shape
                                                                              action_shape=fitness_function.action_spec_shape)

    return model_for_perturb_state_dict


def select(children, n_parents, parents_evaluated, percent_old_population_survive):
    print(f'{children=}')
    indices_selected = [c['model_id'] for c in children]

    n_elites = int(percent_old_population_survive * n_parents)
    fitnesses_parents = [p['fitnesses'][-1] for p in parents_evaluated]
    indices_sorted = np.argsort(fitnesses_parents)
    indices_elites = indices_sorted[-n_elites:].tolist()

    # may need to chuck some amount of children
    indices_selected = np.random.permutation(indices_selected)[:n_parents - n_elites].tolist()
    indices_selected += indices_elites
    print(f'{indices_elites=}')
    print(f'{indices_selected=}')

    return indices_selected


def update_population_and_checkpoints(basic_folder, info_per_child, children, indices_selected,
                                      n_parents, next_epoch, parents_evaluated, arch_ss):
    indices_survived_parents = sorted([i for i in indices_selected if i < n_parents])
    indices_survived_children = [i for i in indices_selected if i >= n_parents]
    print(f'{len(indices_survived_children)} children survived: {indices_survived_children}')
    print(f'{indices_survived_parents=}')

    new_population = []
    updated = [False for _ in range(n_parents)]
    # first, add 1 copy of each surviving parent to the new population:
    for i in copy.deepcopy(indices_survived_parents):
        if not updated[i]:
            new_population.append(parents_evaluated[i])
            indices_survived_parents.remove(i)
            updated[i] = True
        # otherwise do nothing

    rl = type(arch_ss) is RlNormActSearchSpace
    child_old_id_to_new_id = {}

    for i in range(n_parents):
        if not updated[i]:
            # parent didn't survive => is replaced with some child
            if len(indices_survived_children) > 0: # first try adding a child
                child = copy.deepcopy(children[indices_survived_children.pop(0) - n_parents])
            else:
                child = copy.deepcopy(parents_evaluated[indices_survived_parents.pop(0)])

            src_file = os.path.join(basic_folder, 'models', f'model_{child["model_id"]}_{next_epoch}')
            dst_file = os.path.join(basic_folder, 'models', f'model_{i}_{next_epoch}')

            print(f'Renumber child {child["model_id"]} to {i}: copy', src_file, dst_file)
            child_old_id_to_new_id[child["model_id"]] = i
            os.remove(dst_file)
            os.rename(src_file, dst_file)

            # renumber workdir
            if rl:
                # need to delete workdir on all of them, because it has to be replaced by a new one
                # (which will be downloaded to worker node in "update_one_model")
                utils.general.ray_run_fun_once_per_node(rmtree_if_exists, os.path.join(basic_folder, 'models', f'workdir_{i}'))
                print(f'replace workdir_{i} with workdir_{child["model_id"]}')
                shutil.copytree(os.path.join(basic_folder, 'models', f'workdir_{child["model_id"]}'),
                                os.path.join(basic_folder, 'models', f'workdir_{i}'))

            # modify history
            if len(indices_survived_children) > 0: # don't do it when replacing by a parent
                for changed_idx in info_per_child[child['model_id']]['ind_changed']:
                    child['history'][changed_idx].append(i)  # add the new id to history of the values that were copied

            child['model_id'] = i  # update model id after the file has been copied
            #                        (because need to know old model id before that)
            new_population.append(child)

            updated[i] = True

    assert len(indices_survived_children) == 0, "all should've been added to the new pop & renumbered"
    return new_population, child_old_id_to_new_id

def update_global_history(basic_folder, child_old_id_to_new_id, children, global_history, indices_selected,
                          info_per_child, n_parents, next_epoch, parents_evaluated):
    # 1) finish history of the replaced ones
    for i in range(n_parents):
        if i not in indices_selected:
            for hist_entry in reversed(global_history):
                if hist_entry['model_id'] == i:
                    break
            hist_entry['fitness_final'] = parents_evaluated[i]['fitnesses'][-1]
            hist_entry['epoch_final'] = next_epoch
    # 2) add new ones
    for c in children:
        if c['model_id'] in indices_selected:
            info = info_per_child[c['model_id']]
            new_id = child_old_id_to_new_id[c['model_id']]
            hist_entry = dict(seq_id=len(global_history), model_id=new_id, epoch=next_epoch,
                              parent1=info['parent1'], parent2=info['parent2'], indices=info['ind_changed'],
                              old_values=info['old_values'], new_values=info['new_values'],
                              epoch_final=None, fitness_final=None)
            global_history.append(hist_entry)
    yaml.safe_dump(global_history, open(os.path.join(basic_folder, f'history_{next_epoch}.yml'), 'w'),
                   default_flow_style=None)