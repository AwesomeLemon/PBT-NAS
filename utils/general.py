import logging
import os
import random
import shutil
import sys
import time
import traceback
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import ray
import torch
import yaml
from ray.util import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.optim import AdamW


class LogWriter:
    def __init__(self, log_fun):
        self.log_fun = log_fun
        self.buf = []
        self.is_tqdm_msg_fun = lambda msg: '%|' in msg
        # annoyingly, ray doesn't allow to disable colors in output, and they make logs unreadable, so:
        self.replace_garbage = lambda msg: msg.replace('[2m[36m', '').replace('[0m', '')

    def write(self, msg):
        is_tqdm = self.is_tqdm_msg_fun(msg)
        has_newline = msg.endswith('\n')
        if has_newline or is_tqdm:
            self.buf.append(msg)  # .rstrip('\n'))
            self.log_fun(self.replace_garbage(''.join(self.buf)))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass

    def close(self):
        self.log_fun.close()


def setup_logging(log_path):
    from importlib import reload
    reload(logging)
    logging.StreamHandler.terminator = ''  # don't add new line, I'll do it myself; this line affects both handlers
    stream_handler = logging.StreamHandler(sys.__stdout__)
    file_handler = logging.FileHandler(log_path, mode='a')
    # don't want a bazillion tqdm lines in the log:
    # file_handler.filter = lambda record: '%|' not in record.msg or '100%|' in record.msg
    file_handler.filter = lambda record: '[A' not in record.msg and ('%|' not in record.msg or '100%|' in record.msg)
    handlers = [
        file_handler,
        stream_handler]
    logging.basicConfig(level=logging.INFO,
                        # format='%(asctime)s %(message)s',
                        # https://docs.python.org/3/library/logging.html#logrecord-attributes
                        # https://docs.python.org/3/library/logging.html#logging.Formatter
                        # format='%(process)d %(message)s',
                        format='%(message)s',
                        handlers=handlers,
                        datefmt='%H:%M')
    sys.stdout = LogWriter(logging.info)
    sys.stderr = LogWriter(logging.error)


def adjust_optimizer_settings(optimizer, lr, wd=None):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        if wd is not None:
            param_group['weight_decay'] = wd

    return optimizer


def worker_init_fn(worker_id):
    # print(worker_id, np.random.get_state()[1][0] + worker_id)
    torch_seed = torch.initial_seed() + worker_id
    torch_seed = torch_seed % 2 ** 30

    # print(torch_seed, worker_id)
    random.seed(torch_seed)
    np.random.seed(torch_seed)


def set_random_seeds(seed=42):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def optimizer_to(optim, device):
    if not (type(optim) is dict):
        all_values = [optim.state.values()]
    else:  # in gan, optim is a dict of optims
        all_values = [o.state.values() for o in optim.values()]
    for values in all_values:
        for param in values:
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)


def seq_to_str(ar):
    return ','.join([str(t) for t in ar])


class MyAdamW(AdamW):
    r"""My modification of AdamW: remove state if the shape doesn't match
    (this way, the state for unchanged parts of the network is kept)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        super(MyAdamW, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def __setstate__(self, state):
        super(MyAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

            for p in group['params']:

                state = self.state[p]
                if len(state) == 0:
                    continue

                if state['exp_avg'].shape != p.shape:
                    # assert state['exp_avg'].shape == torch.Size([17]), f"{state['exp_avg'].shape=} {p.shape=}"
                    print('Deleted mismatching state without checking')
                    del self.state[p]


@ray.remote(num_cpus=4, num_gpus=1, max_calls=1, resources={'head_node': 1})
def run_ray_FID_or_Inception(fitness_function, fun_name, *args):
    assert fun_name in ['evaluate_final_gan_FID', 'evaluate_final_gan_inception']
    set_non_determentistic()  # for dilated convolutions
    return getattr(fitness_function, fun_name)(*args)


def evaluate_final_gan_ray(basic_folder, fitness_function_ref, args, soups, ckpt_lbls=['ever', 'last']):
    # compute FID-50k for best & last on train & test
    def evaluate_model(ckpt_lbl, prefix):
        ckpt_name = f'{prefix}{ckpt_lbl}_model'
        if not os.path.exists(os.path.join(basic_folder, ckpt_name)):
            print(f'{ckpt_name} does not exist')
            return

        data_dict = yaml.safe_load(open(os.path.join(basic_folder, f'{prefix}{ckpt_lbl}_info.yml')))

        fut_fid_train = run_ray_FID_or_Inception.remote(fitness_function_ref, 'evaluate_final_gan_FID',
                                                        data_dict['pop_member']['hyperparameters'],
                                                        ckpt_name, 50000, 'train')

        fut_best_inception = run_ray_FID_or_Inception.remote(fitness_function_ref, 'evaluate_final_gan_inception',
                                                             data_dict['pop_member']['hyperparameters'], ckpt_name)

        data_dict['pop_member']['fid_train'] = ray.get(fut_fid_train)

        mean, std = ray.get(fut_best_inception)
        data_dict['pop_member']['inception'] = mean
        data_dict['pop_member']['inception_std'] = std

        yaml.safe_dump(data_dict, open(os.path.join(basic_folder, f'{prefix}{ckpt_lbl}_info.yml'), 'w'),
                       default_flow_style=None)

    for soup_type, soup_target in soups:
        for timepoint in ['last', 'ever']:
            try:
                evaluate_model('soup_' + get_combined_soup_name(soup_type, soup_target, timepoint), '')
            except:
                print('Encountered exception, most likely because soup does not exist => ignore')
                print(traceback.format_exc())

    for ckpt_lbl in ckpt_lbls:
        evaluate_model(ckpt_lbl, 'best_')


def set_non_determentistic():
    # in a separate function because otherwise Ray complains about being unable to pickle a Cudnn object
    torch.backends.cudnn.deterministic = False


def try_load_checkpoint(folder, checkpoint_name):
    checkpoint_full_name = '%s/%s' % (folder, checkpoint_name)

    if not os.path.exists(checkpoint_full_name):
        print(f'Checkpoint for model {checkpoint_name} not found')
        return None

    checkpoint = torch.load(checkpoint_full_name, map_location='cpu')

    return checkpoint


@ray.remote(num_cpus=1, max_calls=1)  # this num_cpus shouldn't be changed!
def ray_wrap(fun, *args):
    fun(*args)


def ray_run_fun_once_per_node(fun, *args):
    # st = time.time()
    bundles = [{"CPU": 1} for _ in ray.nodes()]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        pg = placement_group(bundles, "STRICT_SPREAD")
        ray.get(pg.ready())
        tasks = [ray_wrap.options(scheduling_strategy=PlacementGroupSchedulingStrategy(pg)).remote(fun, *args)
                 for i in range(len(ray.nodes()))]
        for t in tasks:
            ray.get(t)
        remove_placement_group(pg)
    # print(f'Time in ray_run_fun_once_per_node: {time.time() - st:.2f}')


def print_individual(individual):
    hps = '|'.join(map(str, individual['hyperparameters']))
    print(f"\tmodel_id={individual['model_id']} HPs={hps}")
    if 'history' in individual:
        for k, v in individual['history'].items():
            v = ','.join(map(str, v))
            print(f"{k}:[{v}] ", end='')
        print()
    special_keys = ['model_id', 'hyperparameters', 'history']
    for k, v in individual.items():
        if k in special_keys:
            continue
        if type(v) is list and len(v) > 0 and all(type(v[i]) is float for i in range(len(v))):
            v = ', '.join(map(lambda x: f'{x:.4f}', v))
        print(f"{k}: {v}")


def print_population(population, name):
    print(name)
    for individiual in population:
        print_individual(individiual)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def swap_model_params(model, params_new):
    '''
    sets model to use params_new; returns old parameters of the model
    '''
    params_old = copy_params(model)
    load_params(model, params_new)
    return params_old


@ray.remote(num_cpus=1, max_calls=1, resources={'head_node': 1})
def delete_dir_on_head_node(path):
    if os.path.exists(path):
        print(f'delete {path} on head node')
        shutil.rmtree(path)


def clean_up_dev_shm():
    '''
    for RL with shared replay buffer: in case there are episodes in shared memory, remove them
    '''
    import subprocess
    result = subprocess.run(["find", "/dev/shm/", "-name", '*.npz*', '-delete', '-print'], shell=False,
                            capture_output=True)
    print('Captured output: ', result)


def create_dir(dir_name):
    Path(dir_name).mkdir(exist_ok=True)
    # if the function is too fast, ray seems to ignore the SPREAD_STRICT strategy,
    # and creates the dir on the same node twice, and on one node 0 times.
    time.sleep(2)


def delete_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
        print(f'Deleted {dir_name} {os.path.exists(dir_name)=}')
    # if the function is too fast, ray seems to ignore the SPREAD_STRICT strategy,
    # and creates the dir on the same node twice, and on one node 0 times.
    time.sleep(2)


def rmtree_if_exists(path):
    if not os.path.exists(path):
        return
    try:
        shutil.rmtree(path)
    except OSError:
        pass


def get_combined_soup_name(soup_type, soup_target, timepoint):
    if soup_target == '':
        return soup_type + '_' + timepoint
    return soup_type + '_' + soup_target + '_' + timepoint
