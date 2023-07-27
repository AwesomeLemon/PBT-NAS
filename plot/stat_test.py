from scipy.stats import wilcoxon

from plot.plotting_functions import execute_func_for_all_runs_and_combine, read_actual_fid, return_actual_fid, \
    read_final_score, return_final_score

def get_wilcoxon_p(x, y, alternative):
    print(x)
    print(y)
    return wilcoxon(x, y, alternative=alternative).pvalue

if __name__ == '__main__':
    print('GAN: PBT-NAS FID < Random')
    pbtnas_c10_easy = execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_Gan',
                                                       read_actual_fid, func_combine=return_actual_fid)
    pbtnas_c10_hard = execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_GanHard',
                                                       read_actual_fid, func_combine=return_actual_fid)
    pbtnas_stl_hard = execute_func_for_all_runs_and_combine('stl10advnas_PBTNAS_pop24',
                                                       read_actual_fid, func_combine=return_actual_fid)

    random_c10_easy = execute_func_for_all_runs_and_combine('cifar10advnas_RandomSearch_pop24_Gan',
                                                       read_actual_fid, func_combine=return_actual_fid)
    random_c10_hard = execute_func_for_all_runs_and_combine('cifar10advnas_RandomSearch_pop24_GanHard',
                                                       read_actual_fid, func_combine=return_actual_fid)
    random_stl_hard = execute_func_for_all_runs_and_combine('stl10advnas_RandomSearch_pop24',
                                                       read_actual_fid, func_combine=return_actual_fid)
    print(get_wilcoxon_p(pbtnas_c10_easy + pbtnas_c10_hard + pbtnas_stl_hard,
                         random_c10_easy + random_c10_hard + random_stl_hard, alternative='less'))

    print('GAN: PBT-NAS FID < SEARL')
    pbtnas_c10_easy = execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_Gan',
                                                       read_actual_fid, func_combine=return_actual_fid)
    pbtnas_c10_hard = execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_GanHard',
                                                       read_actual_fid, func_combine=return_actual_fid)
    pbtnas_stl_hard = execute_func_for_all_runs_and_combine('stl10advnas_PBTNAS_pop24',
                                                       read_actual_fid, func_combine=return_actual_fid)

    searl_c10_easy = execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_Gan_mutation',
                                                           read_actual_fid, func_combine=return_actual_fid)
    searl_c10_hard = execute_func_for_all_runs_and_combine('cifar10advnas_PBTNAS_pop24_GanHard_mutation',
                                                           read_actual_fid, func_combine=return_actual_fid)
    searl_stl_hard = execute_func_for_all_runs_and_combine('stl10advnas_PBTNAS_pop24_mutation',
                                                           read_actual_fid, func_combine=return_actual_fid)
    print(get_wilcoxon_p(pbtnas_c10_easy + pbtnas_c10_hard + pbtnas_stl_hard,
                         searl_c10_easy + searl_c10_hard + searl_stl_hard, alternative='less'))

    print('RL: PBT-NAS Score > Random')
    pbtnas_quadrun = execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop12',
                                                       read_final_score, func_combine=return_final_score)
    pbtnas_walker = execute_func_for_all_runs_and_combine('walker_run_PBTNAS_pop12',
                                                       read_final_score, func_combine=return_final_score)
    pbtnas_humanoid = execute_func_for_all_runs_and_combine('humanoid_run_PBTNAS_pop12',
                                                       read_final_score, func_combine=return_final_score)

    random_quadrun = execute_func_for_all_runs_and_combine('quadruped_run_RandomSearch_pop12',
                                                       read_final_score, func_combine=return_final_score)
    random_walker = execute_func_for_all_runs_and_combine('walker_run_RandomSearch_pop12',
                                                       read_final_score, func_combine=return_final_score)
    random_humanoid = execute_func_for_all_runs_and_combine('humanoid_run_RandomSearch_pop12',
                                                       read_final_score, func_combine=return_final_score)
    print(get_wilcoxon_p(pbtnas_quadrun + pbtnas_walker + pbtnas_humanoid,
                         random_quadrun + random_walker + random_humanoid, alternative='greater'))

    print('RL: PBT-NAS Score > SEARL')
    pbtnas_quadrun = execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop12',
                                                       read_final_score, func_combine=return_final_score)
    pbtnas_walker = execute_func_for_all_runs_and_combine('walker_run_PBTNAS_pop12',
                                                       read_final_score, func_combine=return_final_score)
    pbtnas_humanoid = execute_func_for_all_runs_and_combine('humanoid_run_PBTNAS_pop12',
                                                       read_final_score, func_combine=return_final_score)

    searl_quadrun = execute_func_for_all_runs_and_combine('quadruped_run_PBTNAS_pop12_mutation',
                                                       read_final_score, func_combine=return_final_score)
    searl_walker = execute_func_for_all_runs_and_combine('walker_run_PBTNAS_pop12_mutation',
                                                       read_final_score, func_combine=return_final_score)
    searl_humanoid = execute_func_for_all_runs_and_combine('humanoid_run_PBTNAS_pop12_mutation',
                                                       read_final_score, func_combine=return_final_score)
    print(get_wilcoxon_p(pbtnas_quadrun + pbtnas_walker + pbtnas_humanoid,
                         searl_quadrun + searl_walker + searl_humanoid, alternative='greater'))