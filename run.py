import argparse
import cv2
import datetime
import os
import ray
import shutil
import time
import traceback
import yaml
from copy import deepcopy
from types import SimpleNamespace

import utils
import utils.general
from search.algorithms.ea import run_EA
from search.algorithms.random_search import run_baseline
from utils.general import setup_logging, clean_up_dev_shm, create_dir


def run_many(configs_all):
    for config in configs_all:
        try:
            run(config)
        finally:
            seconds_to_sleep = 30
            print(f'Sleeping for {seconds_to_sleep} seconds to give Ray time to cancel all tasks')
            time.sleep(seconds_to_sleep)
            utils.general.ray_run_fun_once_per_node(clean_up_dev_shm)

    print(datetime.datetime.now())


def run(config):
    try:
        args = SimpleNamespace(**config)

        # create directories
        log_folder = config['logs_path']
        utils.general.ray_run_fun_once_per_node(create_dir, log_folder)
        run_folder = args.out_name_template.format(**config)
        run_folder = os.path.join(log_folder, run_folder)

        utils.general.ray_run_fun_once_per_node(create_dir, run_folder)
        seed_folder = os.path.join(run_folder, str(config['i_seed']))

        utils.general.ray_run_fun_once_per_node(create_dir, seed_folder)

        setup_logging(os.path.join(seed_folder, '_log.txt'))
        print(datetime.datetime.now())

        shutil.copy(config_path, seed_folder)
        args.folder = seed_folder
        print(args)

        if args.algorithm == 'RandomSearch':
            run_baseline(args)
        elif args.algorithm == 'PBTNAS':
            run_EA(args)
        else:
            raise NotImplementedError(args.algorithm)

        print(f'{seed_folder=}')
    except Exception as e:
        print(traceback.format_exc())
        print(e)

def create_configs_all(config):
    configs_all = []
    offset = config.get('seed_offset', 0)
    for i_seed in range(offset, offset + config['n_seeds']):
        config_cur = deepcopy(config)
        config_cur['seed'] += i_seed
        config_cur['i_seed'] = i_seed
        configs_all.append(config_cur)
    return configs_all

if __name__ == '__main__':
    cv2.setNumThreads(0)
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--config', default='configs/main/c10_pbtnas.yml', type=str)
    parser.add_argument(f'--slurm', default='no', type=str)

    parsed_args = parser.parse_args()
    config_path = parsed_args.config
    config = yaml.safe_load(open(config_path))

    configs_all = create_configs_all(config)

    if_slurm = parsed_args.slurm == 'yes' # should've used a boolean flag instead of yes/no
    if if_slurm:
        # special scripts are used to set up Ray on a Slurm cluster, in Python only need to run ray.init
        ray.init()
    else:
        tmp_dir = '/tmp/ray'
        try:
            # try to attach to an already running ray cluster
            ray.init(address='auto', _temp_dir=tmp_dir)
        except ConnectionError:
            print('Starting ray locally')
            resources_dict = {'head_node': 1000} # this resource is used to when a function has to run on the head node
            resources_dict.update({str(i): 1 for i in range(36)}) # these resources are used in RL when buffer
            #         is not shared to tie each model to a machine, so that its replay buffer wouldn't have to be synced
            ray.init(_temp_dir=tmp_dir, resources=resources_dict)

    run_many(configs_all)
