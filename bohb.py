from types import SimpleNamespace

import cv2
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

import argparse
import os
import shutil
from pathlib import Path

import torch
import yaml
from ray import tune
from ray.tune.experiment.trial import Trial

from run import create_configs_all
from search.algorithms.common_utils import init_run, delete_old_model_files
from search.hyperparameters import Hyperparameters
from search_space import get_arch_search_space
from utils.general import set_random_seeds, evaluate_final_gan_ray

class ArchitectureTrainable(tune.Trainable):
    def setup(self, config):
        print('In setup!')
        print(f'{self._iteration=}')
        self.config = config

        trial_dirname = trial_dirname_creator_from_ray_config(config)

        config_my = config['config_my']

        self.trial_path = os.path.join(config_my['logs_path'], config_my['experiment_name'], trial_dirname)

        args = SimpleNamespace(**config_my)
        args.folder = self.trial_path

        _, _, _, _, _, _, _, _, _, _, _, _, EPOCHS_STEP, _, _, _, _, _, _, _, _, _, fitness_function = init_run(args)

        self.epoch = 0
        self.EPOCHS_STEP = EPOCHS_STEP
        self.best_fitness = -np.inf
        self.if_rl = 'rl_general' in config_my

        def config_to_encoded_solution():
            encoded = []
            for i in range(len(config) - 1):
                encoded.append(int(config[f'x{i}']))
            return encoded

        self.encoded_solution = config_to_encoded_solution()
        self.fitness_function = fitness_function
        self.seed = args.seed

    def step(self):
        # seed should be different per model, but now there's no index anymore => replace with hash of experiment name
        cur_seed = self.seed * 100000 + self.epoch * 1000 + np.abs(hash(self.trial_path)) % 1000
        model_id = 0 # each trial == 1 folder == 1 model => always index 0
        fitness_args = (self.encoded_solution, model_id, self.epoch)
        fitness_kwargs = dict(save_at_epoch_0=True, seed=cur_seed)

        delete_old_model_files(self.trial_path, self.epoch)

        fitness_value = self.fitness_function.fitness(*fitness_args, **fitness_kwargs)
        # may want to remove old checkpoint afterwards

        self.epoch += self.EPOCHS_STEP

        if not self.if_rl:
            if fitness_value > self.best_fitness:
                self.best_fitness = fitness_value
                info_best = {'epoch': self.epoch, 'pop_member': {'hyperparameters': self.encoded_solution, 'fitness': fitness_value}}
                yaml.safe_dump(info_best, open(os.path.join(self.trial_path, 'best_ever_info.yml'), 'w'),
                               default_flow_style=None)
                copy_from = os.path.join(self.trial_path, 'models', f'model_{model_id}_{self.epoch}')
                shutil.copy(copy_from, os.path.join(self.trial_path, 'best_ever_model'))
        else:
            self.best_fitness = fitness_value
            info_last = {'epoch': self.epoch, 'pop_member': {'hyperparameters': self.encoded_solution, 'fitness': fitness_value}}
            yaml.safe_dump(info_last, open(os.path.join(self.trial_path, 'best_last_info.yml'), 'w'),
                           default_flow_style=None)
            copy_from = os.path.join(self.trial_path, 'models', f'model_{model_id}_{self.epoch}')
            shutil.copy(copy_from, os.path.join(self.trial_path, 'best_last_model'))

        res = {"fitness": fitness_value, "epoch": self.epoch}
        print(f'{res=}')
        return res

    def save_checkpoint(self, tmp_checkpoint_dir):
        # don't need to do anything, I ignore ray's checkpoint dirs because my code saves everything normally into
        # the trial dir. However, I still need this function to make ray aware that I have checkpoints
        # and thus continuation from them is possible.
        checkpoint_path = os.path.join(self.trial_path, "info.pth")
        torch.save({'epoch': self.epoch, 'best_fitness': self.best_fitness}, checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        # ignore tmp_checkpoint_dir
        loaded = torch.load(os.path.join(self.trial_path, f'info.pth'))
        print(f'{loaded=}')
        self.epoch = loaded['epoch']
        self.best_fitness = loaded['best_fitness']

    def final_eval(self):
        if not self.if_rl:
            evaluate_final_gan_ray(self.trial_path, self.fitness_function, None, [], ['ever'])
            shutil.copy(os.path.join(self.trial_path, 'best_ever_info.yml'),
                        os.path.join(Path(self.trial_path).parent.absolute(), 'best_ever_info.yml'))
            shutil.copy(os.path.join(self.trial_path, 'best_ever_model'),
                        os.path.join(Path(self.trial_path).parent.absolute(), 'best_ever_model'))
        else:
            shutil.copy(os.path.join(self.trial_path, 'best_last_info.yml'),
                        os.path.join(Path(self.trial_path).parent.absolute(), 'best_last_info.yml'))
            shutil.copy(os.path.join(self.trial_path, 'best_last_model'),
                        os.path.join(Path(self.trial_path).parent.absolute(), 'best_last_model'))

def trial_dirname_creator_from_ray_config(config):
    name = ''
    for i in range(len(config) - 1):
        name += str(config[f'x{i}'])
    return name

def trial_dirname_creator(trial: Trial):
    return trial_dirname_creator_from_ray_config(trial.config)

if __name__ == '__main__':
    cv2.setNumThreads(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(f'--config', default='configs/ablation/bohb/c10_bohb.yml', type=str)
    parser.add_argument(f'--slurm', default='no', type=str)
    parsed_args = parser.parse_args()

    if_slurm = parsed_args.slurm == 'yes'  # should've used a boolean flag instead of yes/no
    if if_slurm:
        # special scripts are used to set up Ray on a Slurm cluster, in Python only need to run ray.init
        ray.init()
    else:
        tmp_dir = '/tmp/ray'
        # try to attach to an already running ray cluster
        ray.init(address='auto', _temp_dir=tmp_dir)

    config_path = parsed_args.config
    config_my = yaml.safe_load(open(config_path))

    assert config_my['n_seeds'] == 1, 'only 1 seed is supported for now'
    configs_all = create_configs_all(config_my)
    config_my = configs_all[0]

    config_my['experiment_name'] = config_my['out_name_template'].format(**config_my)

    # create search space
    args = SimpleNamespace(**config_my)
    set_random_seeds(args.seed)
    arch_ss = get_arch_search_space(**vars(args))
    hyperparameters = Hyperparameters(arch_ss, **vars(args))
    alphabet_array = np.array(hyperparameters.alphabet)

    config = {f'x{i}': tune.choice(list(range(alphabet_array[i]))) for i in range(len(alphabet_array))}
    #                    added 'x' to param name because otherwise ray converts dict to list & everything fails.
    config['config_my'] = config_my

    algo = TuneBOHB(seed=args.seed)
    algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=6)
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=30,
    )

    num_samples = args.num_samples # empirically determined to be equivalent to 24 * 30 total steps (10 epochs per step)

    result = tune.run(
        ArchitectureTrainable,
        resources_per_trial={"cpu": 2, "gpu": 0.5},
        config=config,
        metric="fitness",
        mode="max",
        search_alg=algo,
        num_samples=num_samples,
        scheduler=scheduler,
        local_dir=config_my['logs_path'],
        # resume=True,
        name=config_my['experiment_name'],
        keep_checkpoints_num=1,
        progress_reporter=ray.tune.CLIReporter(max_report_frequency=120, max_progress_rows=50, max_error_rows=50),
        log_to_file=("my_stdout.log", "my_stderr.log"),
        trial_dirname_creator=trial_dirname_creator,
        stop={"training_iteration": 30},
        checkpoint_freq=1
    )

    if_rl = 'rl_general' in config_my
    mode_which_best = "last" if if_rl else "all"
    best_trial = result.get_best_trial("fitness", "max", mode_which_best)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final fitness: {}".format(best_trial.last_result["fitness"]))

    trainable_for_final_eval = ArchitectureTrainable(config=best_trial.config)
    trainable_for_final_eval.final_eval()