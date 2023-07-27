import copy
import os
import pickle
import shutil
from collections import defaultdict

import numpy as np
import ray
import torch
import yaml

from utils.general import run_ray_FID_or_Inception, try_load_checkpoint, get_combined_soup_name


def cook_model_soup_greedy_ray(basic_folder, fitness_function, args, soup_type, soup_target, timepoint):
    assert timepoint in ['ever', 'last']
    epoch = yaml.safe_load(open(os.path.join(basic_folder, f'best_{timepoint}_info.yml')))['epoch']
    pop = pickle.load(open(os.path.join(basic_folder, f'population_{epoch}.pkl'), 'rb'))
    model_dir = 'models'
    if timepoint == 'ever':
        model_dir = 'models_best_iter'

    # get ids of models with the same architecture
    if soup_type == 'frequent':
        sol_encoding, ids_same_sol, id_to_fitness = get_most_frequent_solution_info(pop)
    else:
        assert soup_type == 'best'
        sol_encoding, ids_same_sol, id_to_fitness = get_best_solution_info(pop)

    # try adding models greedily

    soup_sd = None  # 'sd' means 'state_dict'
    soup_avg_gen_sd = None
    soup_ids = []

    soup_history = []

    for model_id in ids_same_sol:
        ckpt = try_load_checkpoint(fitness_function.folder, f'{model_dir}/model_{model_id}_{epoch}')
        model_state_dict = ckpt['model_state_dict']

        ignored_ks = []

        soup_ids.append(model_id)
        if soup_sd is None:
            soup_sd = model_state_dict
            soup_fitness = id_to_fitness[model_id]
        else:
            soup_sd_copy = copy.deepcopy(soup_sd)
            for k, v in model_state_dict.items():
                if (k[-3:] == '._u') or (k[-3:] == '._v'):  # these are singular vectors in the spectral normalization, averaging them is a bad idea
                    continue

                if (soup_target != '') and (soup_target not in k):
                    ignored_ks.append(k)
                    continue

                soup_sd_copy[k] += v

        print(f'{ignored_ks=}')

        if 'avg_gen_state_dict' in ckpt:
            avg_gen_sd = ckpt['avg_gen_state_dict']
            if soup_avg_gen_sd is None:
                soup_avg_gen_sd = avg_gen_sd
            else:
                soup_avg_gen_sd_copy = copy.deepcopy(soup_avg_gen_sd)
                for k, v in avg_gen_sd.items():
                    soup_avg_gen_sd_copy[k] += v

        if len(soup_ids) == 1:
            print(f'{soup_ids=} [First]')
            continue

        soup_sd_copy2 = copy.deepcopy(soup_sd_copy)
        for k in soup_sd_copy.keys():
            if (k[-3:] == '._u') or (k[-3:] == '._v'):
                continue
            if k in ignored_ks:
                continue
            if soup_sd_copy[k].dtype is torch.long:
                soup_sd_copy[k] //= len(soup_ids)
            else:
                soup_sd_copy[k] /= len(soup_ids)

        new_ckpt = copy.deepcopy(ckpt)
        new_ckpt['model_state_dict'] = soup_sd_copy
        if 'avg_gen_state_dict' in ckpt:
            soup_avg_gen_sd_copy2 = copy.deepcopy(soup_avg_gen_sd_copy)
            for k in soup_avg_gen_sd_copy.keys():
                if soup_avg_gen_sd_copy[k].dtype is torch.long:
                    soup_avg_gen_sd_copy[k] //= len(soup_ids)
                else:
                    soup_avg_gen_sd_copy[k] /= len(soup_ids)

            new_ckpt['avg_gen_state_dict'] = soup_avg_gen_sd_copy

        new_ckpt_path = os.path.join(basic_folder, 'soup_greedy_tmp')
        torch.save(new_ckpt, new_ckpt_path)

        # evaluate
        if args.model_parameters['task'] == 'generation':
            fut_fitness = run_ray_FID_or_Inception.remote(fitness_function, 'evaluate_final_gan_FID',
                                                          sol_encoding,
                                                          'soup_greedy_tmp', 5000, 'train')
        else:
            eval_env_seed = np.random.randint(10 ** 6,
                                              10 ** 9)  # need to generate randomly in order not to overfit to a single seed
            fut_fitness = run_method_of_fitness_fun_on_head.remote(fitness_function, 'fitness_no_train', sol_encoding,
                                                   'soup_greedy_tmp', eval_env_seed, args.max_epochs)
        fitness_cur = ray.get(fut_fitness)

        soup_history.append({'soup_ids': copy.deepcopy(soup_ids), 'accepted': bool(fitness_cur > soup_fitness),
                             'candidate': model_id, 'fitness_soup': float(fitness_cur)})

        if fitness_cur > soup_fitness:
            soup_fitness = fitness_cur
            soup_sd = soup_sd_copy2  # the version in memory should be the sum, without division
            if 'avg_gen_state_dict' in ckpt:
                soup_avg_gen_sd = soup_avg_gen_sd_copy2
            shutil.copy(new_ckpt_path, os.path.join(basic_folder, 'soup_greedy'))
            print(f'{soup_ids=} Fitness: {fitness_cur} [Accepted {model_id}]')
        else:
            del soup_ids[-1]
            print(f'{soup_ids=} Fitness: {fitness_cur} [Rejected {model_id}]')

    print(f'Model soup [Greedy]: Fitness: {soup_fitness} {soup_ids=}')

    print('Saving soup info')
    data_dict = yaml.safe_load(open(os.path.join(basic_folder, f'best_{timepoint}_info.yml')))
    data_dict['pop_member'] = {}
    data_dict['pop_member']['hyperparameters'] = [int(x) for x in sol_encoding]
    data_dict['pop_member']['fitness_soup'] = float(soup_fitness)
    data_dict['pop_member']['soup_ids'] = [int(x) for x in soup_ids]
    data_dict['soup_history'] = soup_history

    yml_name = f'soup_{get_combined_soup_name(soup_type, soup_target, timepoint)}_info.yml'
    final_model_name = f'soup_{get_combined_soup_name(soup_type, soup_target, timepoint)}_model'

    yaml.safe_dump(data_dict, open(os.path.join(basic_folder, yml_name), 'w'),
                   default_flow_style=None)

    # full eval if soup has >1 models: save soup info here, full eval together with best_ever|last in another function
    if os.path.exists(os.path.join(basic_folder, 'soup_greedy')):
        shutil.move(os.path.join(basic_folder, 'soup_greedy'), os.path.join(basic_folder, final_model_name))

    if os.path.exists(os.path.join(basic_folder, 'soup_greedy_tmp')):
        os.remove(os.path.join(basic_folder, 'soup_greedy_tmp'))


def get_most_frequent_solution_info(pop):
    solution_counts = defaultdict(lambda: 0)
    solution_to_ids = defaultdict(lambda: [])
    id_to_fitness = {}
    for solution in pop:
        solution_counts[tuple(solution['hyperparameters'])] += 1
        solution_to_ids[tuple(solution['hyperparameters'])].append(solution['model_id'])
        id_to_fitness[solution['model_id']] = solution['fitnesses'][-1]
    most_freq_sol_cnt, most_freq_sol_ids, most_freq_sol = 0, None, None
    for sol, sol_cnt in solution_counts.items():
        if sol_cnt > most_freq_sol_cnt:
            most_freq_sol_cnt = sol_cnt
            most_freq_sol_ids = solution_to_ids[sol]
            most_freq_sol = sol
    # sort by fitness
    most_freq_sol_ids.sort(key=lambda id: -id_to_fitness[id])
    return most_freq_sol, most_freq_sol_ids, id_to_fitness


def get_best_solution_info(pop):
    '''
    finds all the solutions that have the same architecture as the best solution (the best solution itself is included)
    '''
    solution_to_fitness = defaultdict(lambda: -1e10)
    solution_to_ids = defaultdict(lambda: [])
    id_to_fitness = {}
    for solution in pop:
        solution_to_fitness[tuple(solution['hyperparameters'])] = max(
            solution_to_fitness[tuple(solution['hyperparameters'])], solution['fitnesses'][-1])
        solution_to_ids[tuple(solution['hyperparameters'])].append(solution['model_id'])
        id_to_fitness[solution['model_id']] = solution['fitnesses'][-1]
    best_fitness, best_sol_ids, best_sol = -1e10, None, None
    for sol, sol_fitness in solution_to_fitness.items():
        if sol_fitness > best_fitness:
            best_fitness = sol_fitness
            best_sol_ids = solution_to_ids[sol]
            best_sol = sol
    # sort by fitness
    best_sol_ids.sort(key=lambda id: -id_to_fitness[id])
    return best_sol, best_sol_ids, id_to_fitness


@ray.remote(num_cpus=4, num_gpus=1, max_calls=1, resources={'head_node': 1})
def run_method_of_fitness_fun_on_head(fitness_function, method_name, *args):
    return getattr(fitness_function, method_name)(*args)