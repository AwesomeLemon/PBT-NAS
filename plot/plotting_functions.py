import itertools
from collections import defaultdict

import re

import os

import glob

import matplotlib
import yaml
from matplotlib import pyplot as plt

import pickle
import numpy as np

LOGS_PATH = '/export/scratch3/aleksand/PBT-NAS/logs_final'  # set to your path!


def execute_func_for_all_runs_and_combine(experiment_name, func, func_combine=None, **kwargs):
    experiment_path = os.path.join(LOGS_PATH, experiment_name)
    seed_to_result = defaultdict(dict)
    target_runs = kwargs.get('target_runs', None)
    for run_folder in sorted(os.scandir(experiment_path), key=lambda ent: int(ent.name) if ent.is_dir() else -1):
        if not run_folder.is_dir():
            continue
        run_idx = int(run_folder.name)
        if target_runs is not None and run_idx not in target_runs:
            continue
        run_path = run_folder.path  # os.path.join(experiment_path, run_folder, str(run_idx))
        out = func(run_path, run_idx=run_idx, experiment_name=experiment_name, **kwargs)
        seed_to_result[run_idx] = out

    if func_combine:
        return func_combine(experiment_path, seed_to_result, experiment_name=experiment_name, **kwargs)

    return seed_to_result


def execute_func_for_many_experiments_and_combine(experiment_names, func, func_combine=None, **kwargs):
    global LOGS_PATH
    LOGS_PATHS_many = [LOGS_PATH]
    exp_to_seed_to_result = {}
    experiment_paths = []
    for i_experiment, experiment_name in enumerate(experiment_names):
        found_experiment = False
        for LOGS_PATH in LOGS_PATHS_many:
            experiment_path = os.path.join(LOGS_PATH, experiment_name)
            if os.path.exists(experiment_path):
                found_experiment = True
                break
        if not found_experiment:
            raise FileNotFoundError(experiment_name)
        experiment_paths.append(experiment_path)
        seed_to_result = defaultdict(dict)
        target_runs = kwargs.get('target_runs', None)
        for run_folder in os.scandir(experiment_path):
            if not run_folder.is_dir():
                continue
            try:
                run_idx = int(run_folder.name)
            except:  # there are other subfolders apart from the run subfolders
                continue
            if target_runs is not None and run_idx not in target_runs:
                continue
            run_path = run_folder.path  # os.path.join(experiment_path, run_folder, str(run_idx))
            out = func(run_path, run_idx=run_idx, experiment_name=experiment_name, experiment_idx=i_experiment,
                       **kwargs)
            seed_to_result[run_idx] = out
        exp_to_seed_to_result[experiment_name] = seed_to_result

    if func_combine:
        return func_combine(experiment_names, experiment_paths, exp_to_seed_to_result, **kwargs)

    return exp_to_seed_to_result

def get_metric_all_epochs(run_path, **kwargs):
    pop_paths = glob.glob(os.path.join(run_path, "population_*.pkl"))
    re_int = re.compile(r'\d+')
    pop_paths = list(sorted(pop_paths, key=lambda p: int(re_int.findall(p)[-1])))
    epochs = [int(re_int.findall(p)[-1]) for p in pop_paths]

    metric_name = kwargs.get('metric')
    metrics = []

    for e, p in zip(epochs, pop_paths):
        metrics_cur_epoch = []
        pop = pickle.load(open(p, 'rb'))
        for solution in pop:
            metric = solution[metric_name][-1]
            metrics_cur_epoch.append(metric)
        metrics.append(metrics_cur_epoch)

    return epochs, metrics


def metric_over_time_many_experiments_single_plot(experiment_names, experiment_paths, exp_to_seed_to_result, **kwargs):
    seed_to_exp_to_res = exp2seed_to_seed2exp(exp_to_seed_to_result)
    seeds = list(seed_to_exp_to_res.keys())
    exp_names_pretty = kwargs['exp_names_pretty']
    metric_name = kwargs.get('metric')
    func_aggregate = kwargs.get('func_aggregate', np.max)
    color_cycler = kwargs.get('color_cycler', itertools.cycle(matplotlib.rcParams['axes.prop_cycle']))
    plot_std_or_separate = kwargs.get('plot_std_or_separate', 'std')
    plot_monotonous = kwargs.get('monotonous', False)
    print_best_instead_of_last = kwargs.get('print_best', False)
    legend_inside = kwargs.get('legend_inside', False)
    marker_style = kwargs.get('marker_style', '-o')
    value_to_add_at_0 = kwargs.get('value_to_add_at_0', None)
    epoch_to_frames_mul = kwargs.get('epoch_to_frames_mul', None)

    # print perf over seeds
    for i_exp_name, seed_to_result in enumerate(exp_to_seed_to_result.values()):
        results_per_seed = []
        for run_result_all_epochs in seed_to_result.values():
            if not print_best_instead_of_last:
                run_result = func_aggregate(run_result_all_epochs[1][-1])
            else:
                run_result = func_aggregate(run_result_all_epochs[1])

            results_per_seed.append(run_result)
        exp_name_prettier = exp_names_pretty[i_exp_name].replace("\n", "_")
        print(
            f'{exp_name_prettier} {np.mean(results_per_seed):.2f} ± {np.std(results_per_seed):.2f} [{len(results_per_seed)} seeds]')

    # plot
    plt.figure(figsize=kwargs.get('figsize', (8, 4)))
    # plt.figure(figsize=kwargs.get('figsize', (6.2, 3.4)))
    # plt.figure(figsize=(5, 4))
    # plt.figure(figsize=(6, 6))
    # plt.figure(figsize=(8, 6))
    # plt.figure(figsize=(8, 10))

    if plot_std_or_separate == 'std':
        for i_exp_name, seed_to_result in enumerate(exp_to_seed_to_result.values()):
            results_per_seed = []
            for run_result_all_epochs in seed_to_result.values():
                aggregated = [func_aggregate(x) for x in run_result_all_epochs[1]]
                if plot_monotonous:
                    aggregated = [func_aggregate(aggregated[:i + 1]) for i in range(len(aggregated))]
                results_per_seed.append(aggregated)
            results_per_seed = np.array(results_per_seed)
            mean = np.mean(results_per_seed, axis=0)
            std = np.std(results_per_seed, axis=0)
            cur_color = next(color_cycler)['color']
            x_values = run_result_all_epochs[0]
            mean_sub_std, mean_add_std = mean - std, mean + std
            if value_to_add_at_0 is not None:
                x_values = [0] + x_values
                mean = np.concatenate(([value_to_add_at_0], mean))
                mean_sub_std = np.concatenate(([value_to_add_at_0], mean_sub_std))
                mean_add_std = np.concatenate(([value_to_add_at_0], mean_add_std))
            if epoch_to_frames_mul is not None:
                x_values = [x * epoch_to_frames_mul for x in x_values]
            plt.plot(x_values, mean, marker_style, label=exp_names_pretty[i_exp_name], color=cur_color)

            plt.fill_between(x_values, mean_sub_std, mean_add_std, color=cur_color, alpha=0.1)
    elif plot_std_or_separate == 'separate':
        for i_exp_name, seed_to_result in enumerate(exp_to_seed_to_result.values()):
            for seed in sorted(seeds):
                run_result_all_epochs = seed_to_result[seed]
                aggregated = [func_aggregate(x) for x in run_result_all_epochs[1]]
                if plot_monotonous:
                    aggregated = [func_aggregate(aggregated[:i + 1]) for i in range(len(aggregated))]
                plt.plot(run_result_all_epochs[0], aggregated, marker_style,
                         label=exp_names_pretty[i_exp_name] + f' [run {seed}]',
                         color=next(color_cycler)['color'])
    else:
        raise ValueError(f'plot_std_or_separate needs to be "std" or "separate", actual value {plot_std_or_separate}')

    if kwargs.get('show_title', True):
        title = kwargs.get('title', None)
        if title is None:
            title = str(f'{seeds=}')
        if plot_monotonous:
            title += ' [best-up-to-epoch]'
        plt.title(title)
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    if epoch_to_frames_mul is None:
        plt.xlabel('Epoch')
    else:
        plt.xlabel('Total frames')
    plt.ylabel(kwargs.get('metric_name_pretty', metric_name))

    if 'additional_line_ys' in kwargs:
        for line_y, line_lbl in zip(kwargs['additional_line_ys'], kwargs['additional_line_labels']):
            plt.axhline(line_y, label=line_lbl, color=next(color_cycler)['color'])

    if legend_inside:
        plt.legend()
    else:
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.tight_layout()

    for i_exp_name in range(len(experiment_paths)):
        cur_save_path = os.path.join(experiment_paths[i_exp_name],
                                     f'over_time_{metric_name}_combined.{kwargs.get("save_format", "png")}')
        plt.savefig(cur_save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def read_actual_fid(run_path, **kwargs):
    best_info = yaml.safe_load(open(os.path.join(run_path, 'best_ever_info.yml')))
    fid = best_info['pop_member']['fid_train']
    inception = best_info['pop_member']['inception']
    return fid, inception


def print_actual_fid(experiment_path, seed_to_result, **kwargs):
    print(f'{kwargs["experiment_name"]} - {len(seed_to_result)} runs')
    fids = [-v[0] for v in seed_to_result.values()]
    print(f'FID {np.mean(fids):.2f} ± {np.std(fids):.2f}')
    inception_scores = [v[1] for v in seed_to_result.values()]
    print(f'IS {np.mean(inception_scores):.2f} ± {np.std(inception_scores):.2f}')


def return_actual_fid(experiment_path, seed_to_result, **kwargs):
    print(f'{kwargs["experiment_name"]} - {len(seed_to_result)} runs')
    fids = [-kv[1][0] for kv in sorted(seed_to_result.items(), key=lambda kv: kv[0])]
    return fids


def read_soup_fid(run_path, **kwargs):
    soup_name = kwargs['soup_name']
    best_info = yaml.safe_load(open(os.path.join(run_path, f'soup_{soup_name}_info.yml')))
    if 'fid_train' in best_info['pop_member']:
        fid = best_info['pop_member']['fid_train']
        inception = best_info['pop_member']['inception']
    else:
        return -1, -1
    return fid, inception


def read_soup_and_actual_fid(run_path, **kwargs):
    res = {}
    res['fid_soup'], res['is_soup'] = read_soup_fid(run_path, **kwargs)
    res['fid_act'], res['is_act'] = read_actual_fid(run_path, **kwargs)
    return res


def print_soup_diffs(experiment_path, seed_to_result, **kwargs):
    print(f'{kwargs["experiment_name"]} ({kwargs["soup_name"]}) - {len(seed_to_result)} runs')
    fids_soup = [-v['fid_soup'] for v in seed_to_result.values()]
    fids_actual = [-v['fid_act'] for v in seed_to_result.values()]
    diffs_filtered = [fs - fa if fs != 1 else 0 for (fs, fa) in zip(fids_soup, fids_actual)]
    print(f'Soup FID diff stats: {np.mean(diffs_filtered):.2f} ± {np.std(diffs_filtered):.2f}')

    iss_soup = [v['is_soup'] for v in seed_to_result.values()]
    iss_actual = [v['is_act'] for v in seed_to_result.values()]
    diffs_filtered = [fs - fa if fs != -1 else 0 for (fs, fa) in zip(iss_soup, iss_actual)]
    print(f'Soup IS diff stats: {np.mean(diffs_filtered):.2f} ± {np.std(diffs_filtered):.2f}')


def read_final_score(run_path, **kwargs):
    best_info = yaml.safe_load(open(os.path.join(run_path, 'best_last_info.yml')))
    return best_info['pop_member']['fitnesses'][-1]


def return_final_score(experiment_path, seed_to_result, **kwargs):
    print(f'{kwargs["experiment_name"]} - {len(seed_to_result)} runs')
    scores = [kv[1] for kv in sorted(seed_to_result.items(), key=lambda kv: kv[0])]
    return scores


def print_final_score(experiment_path, seed_to_result, **kwargs):
    print(f'{kwargs["experiment_name"]} - {len(seed_to_result)} runs')
    scores = [kv[1] for kv in sorted(seed_to_result.items(), key=lambda kv: kv[0])]
    print(f'Score {np.mean(scores):.0f} ± {np.std(scores):.0f}')


def read_soup_score(run_path, **kwargs):
    soup_name = kwargs['soup_name']
    best_info = yaml.safe_load(open(os.path.join(run_path, f'soup_{soup_name}_info.yml')))
    if (len(best_info['pop_member']['soup_ids']) > 1):
        return best_info['pop_member']['fitness_soup']

    best = -1
    for h in best_info['soup_history']:
        best = max(h['fitness_soup'], best)
    return best


def read_soup_and_final_score(run_path, **kwargs):
    res = {'soup': read_soup_score(run_path, **kwargs),
           'final': read_final_score(run_path, **kwargs)}
    return res


def print_soup_diffs_rl(experiment_path, seed_to_result, **kwargs):
    print(f'{kwargs["experiment_name"]} ({kwargs["soup_name"]}) - {len(seed_to_result)} runs')

    scores_soup = [v['soup'] for v in seed_to_result.values()]
    scores_final = [v['final'] for v in seed_to_result.values()]
    diffs = [s - f if s != -1 else np.nan for (s, f) in
             zip(scores_soup, scores_final)]  # no soup could be tried => don't count that seed
    print(f'Soup score diff stats: {int(np.round(np.nanmean(diffs)))} ± {int(np.round(np.nanstd(diffs)))}')


def stackplot_block_history(run_path, **kwargs):
    pop_paths = glob.glob(os.path.join(run_path, "population_*.pkl"))
    re_int = re.compile(r'\d+')
    pop_paths = list(sorted(pop_paths, key=lambda p: int(re_int.findall(p)[-1])))
    epochs = [int(re_int.findall(p)[-1]) for p in pop_paths]

    origin_to_proportion = defaultdict(lambda: [1 / len(pickle.load(open(pop_paths[0], 'rb')))])

    for e, p in zip(epochs, pop_paths):
        pop = pickle.load(open(p, 'rb'))
        pop_size = len(pop)
        n_genes = len(pop[0]['hyperparameters'])
        cnt = {i: 0 for i in range(pop_size)}
        for solution in pop:
            hist = solution['history']
            for block_hist in hist.values():
                first = block_hist[0]
                cnt[first] += 1

        for k, v in cnt.items():
            origin_to_proportion[k].append(v / (pop_size * n_genes))

    origin_to_proportion = sorted(origin_to_proportion.items(), key=lambda x: x[0])

    origins = [o[0] for o in origin_to_proportion]
    proportions = [o[1] for o in origin_to_proportion]

    epoch_to_frames_mul = kwargs.get('epoch_to_frames_mul', None)
    if epoch_to_frames_mul is not None:
        epochs = [x * epoch_to_frames_mul for x in epochs]

    plt.figure(figsize=kwargs.get('figsize', (6, 6)))

    plt.stackplot([0] + epochs, proportions, labels=origins,
                  # linewidth=0,
                  # edgecolor='none'
                  )
    plt.xlabel('Total frames')
    plt.ylabel('Proportion of layers')
    if kwargs.get('show_title', True):
        plt.title(f'Block history over time: {kwargs["name_pretty"]} (run {kwargs["run_idx"]})')
    plt.tight_layout()
    plt.plot()
    cur_save_path = os.path.join(run_path, f'block_history_over_time_{kwargs["name_pretty"]}_{kwargs["run_idx"]}.pdf')
    plt.savefig(cur_save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


def exp2seed_to_seed2exp(data_dict):
    exp_names = list(data_dict.keys())
    seeds = data_dict[exp_names[0]].keys()
    res = {s: {} for s in seeds}
    for seed in seeds:
        for exp_name in exp_names:
            res[seed][exp_name] = data_dict[exp_name][seed]
    return res
