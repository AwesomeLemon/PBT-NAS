from plot.plotting_functions import execute_func_for_all_runs_and_combine, read_actual_fid, print_actual_fid, \
    read_soup_and_actual_fid, print_soup_diffs

if __name__ == '__main__':
    # Table 1:

    execute_func_for_all_runs_and_combine('stl10advnas_RandomSearch_pop24',
                                          read_actual_fid, func_combine=print_actual_fid)
    execute_func_for_all_runs_and_combine('stl10advnas_PBTNAS_pop24_mutation',
                                          read_actual_fid, func_combine=print_actual_fid)
    execute_func_for_all_runs_and_combine('stl10advnas_PBTNAS_pop24',
                                          read_actual_fid, func_combine=print_actual_fid)

    # Soups:
    execute_func_for_all_runs_and_combine('stl10advnas_PBTNAS_pop24',
                                          read_soup_and_actual_fid, func_combine=print_soup_diffs, soup_name='best_ever')

    # Appendix Table 2:
    execute_func_for_all_runs_and_combine('stl10advnas_RandomSearch_pop24_Gan',
                                          read_actual_fid, func_combine=print_actual_fid)
    execute_func_for_all_runs_and_combine('stl10advnas_PBTNAS_pop24_Gan_mutation',
                                          read_actual_fid, func_combine=print_actual_fid)
    execute_func_for_all_runs_and_combine('stl10advnas_PBTNAS_pop24_Gan',
                                          read_actual_fid, func_combine=print_actual_fid)
