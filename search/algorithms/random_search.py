import ray
import time

import utils
import utils.general
from search.algorithms.common_utils import init_run, delete_old_model_files, evaluate_store_clean, \
    store_population_and_best_weights, train_population, init_best_fitness_ever

def run_baseline(args):
    st = time.time()
    N_MODELS, alphabet_array, basic_folder, population, data_provider, \
    function_create_model, hyperparameters, continue_epoch, arch_ss, rsync_wrapper, if_shared_fs, \
        MAX_EPOCHS, EPOCHS_STEP, task, shrink_perturb_always, if_gen_avg, evaluate_mode, num_cpus, \
        emulate_distributed, n_machines_to_emulate, cleanup_final, rl_unshared_buffer, fitness_function = init_run(args)
    print(f'Init time {time.time() - st:.2f}')

    processes_per_gpu = int(args.processes_per_gpu)
    num_gpus = 1 / processes_per_gpu

    rsync_wrapper_ref = ray.put(rsync_wrapper)

    if evaluate_mode:
        utils.general.evaluate_final_gan_ray(basic_folder, fitness_function, args, [])
        return

    best_fitness_ever = init_best_fitness_ever(basic_folder, continue_epoch)

    for epoch in range(0 if continue_epoch is None else continue_epoch, MAX_EPOCHS, EPOCHS_STEP):
        if if_shared_fs:
            delete_old_model_files(basic_folder, epoch)
        else:
            utils.general.ray_run_fun_once_per_node(delete_old_model_files, basic_folder, epoch)

        next_epoch = epoch + EPOCHS_STEP

        population = train_population(population, 0, args, basic_folder, epoch, fitness_function,
                                      next_epoch, task, rsync_wrapper_ref, num_gpus, num_cpus,
                                      emulate_distributed, n_machines_to_emulate, rl_unshared_buffer)


        utils.general.print_population(population, 'Population after training')

        best_fitness_ever = store_population_and_best_weights(basic_folder, best_fitness_ever, next_epoch,
                                                              population, [], task)

        print(f'{basic_folder}')

    evaluate_store_clean(MAX_EPOCHS, args, basic_folder, cleanup_final, fitness_function, if_shared_fs, population,
                         rsync_wrapper, [], task)