from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from plot.plotting_functions import execute_func_for_many_experiments_and_combine, get_metric_all_epochs, \
    metric_over_time_many_experiments_single_plot, execute_func_for_all_runs_and_combine, read_soup_and_final_score, \
    print_soup_diffs_rl, read_final_score, print_final_score, return_final_score, stackplot_block_history

if __name__ == '__main__':
    plt.style.use('seaborn')
    plt.rcParams.update({'font.size': 36})
    plt.rcParams['axes.grid'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    sns.set(font_scale=1.3)

    from cycler import cycler
    palette = sns.color_palette("colorblind", 10)
    plt.rcParams['axes.prop_cycle'] = cycler(color=palette)

    # Figure 3

    execute_func_for_many_experiments_and_combine([
        'quadruped_run_RandomSearch_pop12',
        'quadruped_run_PBTNAS_pop12_mutation',
        'quadruped_run_PBTNAS_pop12',
    ], get_metric_all_epochs, metric='fitnesses', func_combine=metric_over_time_many_experiments_single_plot,
        func_aggregate=np.max, plot_std_or_separate='std',
        exp_names_pretty=['Random search',
                          'SEARL-like mutation',
                          'PBT-NAS',
                          ],
        target_runs=[0, 1, 2],
        show_title=False,
        legend_inside=True, marker_style='-', figsize=(6, 4),#(4.8, 3.5),
        metric_name_pretty='Score',
        value_to_add_at_0=0, epoch_to_frames_mul=10 ** 4 * 12, save_format='pdf'
    )

    # Soups:
    execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop12',
                                          read_soup_and_final_score, func_combine=print_soup_diffs_rl,
                                          soup_name='best_actor_last')
    execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop12',
                                          read_soup_and_final_score, func_combine=print_soup_diffs_rl,
                                          soup_name='best_encoder_last')
    execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop12',
                                          read_soup_and_final_score, func_combine=print_soup_diffs_rl,
                                          soup_name='best_last')

    # Ablations
    # no new architecture
    execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop12_noNewArch',
                                          read_final_score, func_combine=print_final_score)

    # sh-pe [1, 0]
    execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop12_sh1pe0',
                                          read_final_score, func_combine=print_final_score)

    # sh-pe [0, 1]
    execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop12_sh0pe1',
                                          read_final_score, func_combine=print_final_score)

    # sh-pe in random search
    execute_func_for_all_runs_and_combine('quadruped_run_RandomSearch_pop12_enableShrinkPerturb',
                                          read_final_score, func_combine=print_final_score)

    # scaling box plot
    sns.set(font_scale=1.4)

    pbtnas6 = execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop6',
                                                     read_final_score, func_combine=return_final_score)
    pbtnas12 = execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop12',
                                                       read_final_score, func_combine=return_final_score)
    pbtnas18 = execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop18',
                                                       read_final_score, func_combine=return_final_score)

    random6 = execute_func_for_all_runs_and_combine('quadruped_run_RandomSearch_pop6',
                                                        read_final_score, func_combine=return_final_score)
    random12 = execute_func_for_all_runs_and_combine('quadruped_run_RandomSearch_pop12',
                                                       read_final_score, func_combine=return_final_score)
    random18 = execute_func_for_all_runs_and_combine('quadruped_run_RandomSearch_pop18',
                                                       read_final_score, func_combine=return_final_score)

    pbtnas_means = [np.mean(pbtnas6), np.mean(pbtnas12), np.mean(pbtnas18)]
    pbtnas_std = [np.std(pbtnas6), np.std(pbtnas12), np.std(pbtnas18)]

    random_means = [np.mean(random6), np.mean(random12), np.mean(random18)]
    random_stds = [np.std(random6), np.std(random12), np.std(random18)]

    plt.figure(figsize=(6, 4))
    offset = 0.5
    plt.errorbar([6-offset, 12-offset, 18-offset], random_means, yerr=random_stds, label='Random search', color=palette[0], capsize=10, fmt='o', markersize=10, elinewidth=1, markeredgewidth=1)
    plt.errorbar([6+offset, 12+offset, 18+offset], pbtnas_means, yerr=pbtnas_std, label='PBT-NAS', color=palette[2], capsize=10, fmt='o', markersize=10, elinewidth=1, markeredgewidth=1)

    plt.legend(frameon=True)

    plt.grid(axis='x')

    plt.xticks([6, 12, 18], [6, 12, 18])
    plt.xlabel('Population size')
    plt.ylabel('Score (higher is better)')
    plt.tight_layout()
    plt.plot()
    plt.xlim(4, 20)
    plt.savefig('scale_quadrun.pdf', bbox_inches='tight', pad_inches=0)

    plt.show()

    # exploring final architectures
    palette = sns.color_palette("colorblind", 10)
    palette = ['#555555', 'tab:brown'] + palette
    plt.rcParams['axes.prop_cycle'] = cycler(color=palette)

    execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop12',
                                          stackplot_block_history, name_pretty='Quadruped', show_title=False,
                                          epoch_to_frames_mul=10 ** 4 * 12,
                                          figsize=(6, 4)
                                          )