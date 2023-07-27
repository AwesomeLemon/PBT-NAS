from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from plot.plotting_functions import execute_func_for_all_runs_and_combine, read_actual_fid, print_actual_fid, \
    read_soup_and_actual_fid, print_soup_diffs, return_actual_fid

if __name__ == '__main__':
    plt.style.use('seaborn')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    sns.set(font_scale=1.5)

    from cycler import cycler
    palette = sns.color_palette("colorblind", 10)
    palette[:5] = [palette[2], palette[0], palette[1], palette[4], palette[3]]
    plt.rcParams['axes.prop_cycle'] = cycler(color=palette)

    # Table 1:

    # Random search
    execute_func_for_all_runs_and_combine('cifar10advnas_RandomSearch_pop24_GanHard',
                                          read_actual_fid, func_combine=print_actual_fid)
    execute_func_for_all_runs_and_combine('cifar10advnas_RandomSearch_pop24_Gan',
                                          read_actual_fid, func_combine=print_actual_fid)

    # SEARL-like mutation
    execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_GanHard_mutation',
                                          read_actual_fid, func_combine=print_actual_fid)
    execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_Gan_mutation',
                                          read_actual_fid, func_combine=print_actual_fid)

    # PBT-NAS
    execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_GanHard',
                                          read_actual_fid, func_combine=print_actual_fid)
    execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_Gan',
                                          read_actual_fid, func_combine=print_actual_fid)

    # Soups:

    execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_GanHard',
                                          read_soup_and_actual_fid, func_combine=print_soup_diffs,
                                          soup_name='best_ever')

    # Ablations

    # no new architecture
    execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_GanHard_noNewArch',
                                          read_actual_fid, func_combine=print_actual_fid)

    # sh-pe [1, 0]
    execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_GanHard_sh1pe0',
                                          read_actual_fid, func_combine=print_actual_fid)

    # sh-pe [0, 1]
    execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_GanHard_sh0pe1',
                                          read_actual_fid, func_combine=print_actual_fid)

    # sh-pe in random search
    execute_func_for_all_runs_and_combine('cifar10advnas_RandomSearch_pop24_GanHard_enableShrinkPerturb',
                                          read_actual_fid, func_combine=print_actual_fid)

    # Scaling
    pbtnas12 = execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop12_GanHard',
                                                     read_actual_fid, func_combine=return_actual_fid)
    pbtnas24 = execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_GanHard',
                                                     read_actual_fid, func_combine=return_actual_fid)
    pbtnas36 = execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop36_GanHard',
                                                     read_actual_fid, func_combine=return_actual_fid)

    random12 = execute_func_for_all_runs_and_combine('cifar10advnas_RandomSearch_pop12_GanHard',
                                                     read_actual_fid, func_combine=return_actual_fid)
    random24 = execute_func_for_all_runs_and_combine('cifar10advnas_RandomSearch_pop24_GanHard',
                                                     read_actual_fid, func_combine=return_actual_fid)
    random36 = execute_func_for_all_runs_and_combine('cifar10advnas_RandomSearch_pop36_GanHard',
                                                     read_actual_fid, func_combine=return_actual_fid)

    pbtnas_means = [np.mean(pbtnas12), np.mean(pbtnas24), np.mean(pbtnas36)]
    pbtnas_std = [np.std(pbtnas12), np.std(pbtnas24), np.std(pbtnas36)]

    random_means = [np.mean(random12), np.mean(random24), np.mean(random36)]
    random_stds = [np.std(random12), np.std(random24), np.std(random36)]

    plt.figure(figsize=(6, 4))
    offset = 0.5
    plt.errorbar([12 - offset, 24 - offset, 36 - offset], random_means, yerr=random_stds, label='Random search',
                 color=palette[1], capsize=10, fmt='o', markersize=10, elinewidth=1, markeredgewidth=1)
    plt.errorbar([12 + offset, 24 + offset, 36 + offset], pbtnas_means, yerr=pbtnas_std, label='PBT-NAS',
                 color=palette[0], capsize=10, fmt='o', markersize=10, elinewidth=1, markeredgewidth=1)

    plt.legend(frameon=True)

    plt.grid(axis='x')

    plt.xticks([12, 24, 36], [12, 24, 36])
    plt.xlabel('Population size')
    plt.ylabel('FID (lower is better)')
    plt.tight_layout()
    plt.plot()
    plt.xlim(8, 40)
    plt.savefig('scale_c10.pdf', bbox_inches='tight', pad_inches=0)

    plt.show()