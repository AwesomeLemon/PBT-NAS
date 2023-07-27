from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from plot.plotting_functions import execute_func_for_all_runs_and_combine, \
    execute_func_for_many_experiments_and_combine, get_metric_all_epochs, metric_over_time_many_experiments_single_plot, \
    read_soup_and_final_score, print_soup_diffs_rl

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
        'walker_run_RandomSearch_pop12',
        'walker_run_PBTNAS_pop12_mutation',
        'walker_run_PBTNAS_pop12',
    ], get_metric_all_epochs, metric='fitnesses', func_combine=metric_over_time_many_experiments_single_plot,
        func_aggregate=np.max, plot_std_or_separate='std',
        exp_names_pretty=['Random search',
                          'SEARL-like mutation',
                          'PBT-NAS',
                          ],
        target_runs=[0, 1, 2],
        show_title=False,
        legend_inside=True, marker_style='-', figsize=(6, 4),  # (4.8, 3.5),
        metric_name_pretty='Score',
        value_to_add_at_0=0, epoch_to_frames_mul=10 ** 4 * 12, save_format='pdf'
    )

    # Soup
    execute_func_for_all_runs_and_combine('walker_run_PBTNAS_pop12',
                                          read_soup_and_final_score, func_combine=print_soup_diffs_rl,
                                          soup_name='best_actor_last')
    execute_func_for_all_runs_and_combine('walker_run_PBTNAS_pop12',
                                          read_soup_and_final_score, func_combine=print_soup_diffs_rl,
                                          soup_name='best_encoder_last')
    execute_func_for_all_runs_and_combine('walker_run_PBTNAS_pop12',
                                          read_soup_and_final_score, func_combine=print_soup_diffs_rl,
                                          soup_name='best_last')