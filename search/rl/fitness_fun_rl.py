import subprocess

import gc
import os.path

from dm_control import suite
from pathlib import Path

from types import SimpleNamespace

import torch
import time

import search.algorithms.mix
import utils
import utils.general
from search.rl.train_rl import Workspace

from utils.general import adjust_optimizer_settings, optimizer_to
import traceback


class RlFitness:
    def __init__(self, folder, max_epochs, epoch_step, hyperparameters, function_create_model,
                 args, shrink_perturb_always=None):
        self.folder = folder
        self.max_epochs = max_epochs
        self.epoch_step = epoch_step
        self.hyperparameters = hyperparameters
        self.function_create_model = function_create_model
        self.device = 'cuda'
        self.train_kwargs = {}
        self.shrink_perturb_always = shrink_perturb_always

        self.cfg_rl_general = SimpleNamespace(**args.rl_general)
        self.cfg_rl_agent = SimpleNamespace(**args.model_parameters)

        self.share_replay_buffer = self.cfg_rl_general.replay_buffer_type == 'ram_shared_memory'

        # make the environment here to figure out shapes of inputs/outputs - needed to create the model
        domain, task = self.cfg_rl_general.task_name.split('_', 1)
        domain = dict(cup='ball_in_cup').get(domain, domain)
        train_env = suite.load(domain, task, task_kwargs={'random': 42}, visualize_reward=False)
        # self.observation_spec = copy.deepcopy(train_env.observation_spec())
        # print(f'{self.observation_spec=}')
        self.action_spec_shape = train_env.action_spec().shape

        print(f'{self.action_spec_shape=}')
        if train_env is not None:
            train_env.reset()
            train_env.close()
        del train_env
        gc.collect()

    def fitness(self, encoded_solution, model_id, epoch, save_at_epoch_0=False, seed=None,
                rsync_wrapper_ref=None, emulation_i=None):
        try:
            encoded_solution, net, optimizer = self.create_model_and_optimizer(encoded_solution, f'models/model_{model_id}_{epoch}')

            if self.shrink_perturb_always is not None:
                if epoch != 0:
                    model_for_perturb_state_dict = self.function_create_model(hyperparameters=self.hyperparameters,
                                                                              obs_shape=(9, 84, 84),
                                                                              action_shape=self.action_spec_shape).to(self.device).state_dict()
                    net_state = search.algorithms.mix.shrink_perturb_whole_state(net.state_dict(), self.shrink_perturb_always,
                                                                                 model_for_perturb_state_dict)
                    net.load_state_dict(net_state)
                else:
                    print('skipping shrink-perturb for epoch 0')

            model_workdir = Path(os.path.join(self.folder, f'models/workdir_{model_id}'))
            model_workdir.mkdir(exist_ok=True)
            if self.share_replay_buffer:
                common_workdir = Path(os.path.join(self.folder, f'models/workdir' +
                                                   ('' if emulation_i is None else f'_machine{emulation_i}')))
                common_workdir.mkdir(exist_ok=True)
            else:
                common_workdir = model_workdir

            if save_at_epoch_0 and epoch == 0:
                dict_to_save = {
                    'model_state_dict': net.state_dict(),
                    'hyperparameters': self.hyperparameters,
                    'optimizer_state_dict_encoder': None,
                    'optimizer_state_dict_actor': None,
                    'optimizer_state_dict_critic': None,
                }

                torch.save(dict_to_save, '%s/models/model_%d_%d' % (self.folder, model_id, epoch))

            start_step = epoch * self.cfg_rl_general.frames_in_an_epoch // self.cfg_rl_general.action_repeat
            end_step = (epoch + self.epoch_step) * self.cfg_rl_general.frames_in_an_epoch # this one will be divided by 2 in the Until in train
            print(f'{start_step=} {end_step=}')

            workspace = Workspace(common_workdir, model_workdir, seed, net, optimizer, self.device,
                                  self.cfg_rl_general, self.cfg_rl_agent, start_step, end_step, model_id)
            st = time.time()
            metrics = workspace.train()
            print(f'workspace.train time {time.time() - st:.2f}')

            del workspace
            gc.collect()

            val_score = float(metrics['eval'][-1]['episode_reward'])

            net_state = net.state_dict()

            dict_to_save = {'model_state_dict': net_state, 'hyperparameters': self.hyperparameters,
                            'optimizer_state_dict_encoder': optimizer['encoder'].state_dict(),
                            'optimizer_state_dict_actor': optimizer['actor'].state_dict(),
                            'optimizer_state_dict_critic': optimizer['critic'].state_dict(),
                            }

            torch.save(dict_to_save, '%s/models/model_%d_%d' % (self.folder, model_id, epoch + self.epoch_step))

            if rsync_wrapper_ref is not None:
                # delete_dir_on_head_node was causing trouble somehow => replace ray call with direct ssh invocation
                if not (rsync_wrapper_ref.if_shared_fs or
                        rsync_wrapper_ref.ray_head_node == rsync_wrapper_ref._get_name_this_node()):
                    # ray.get(utils.delete_dir_on_head_node.remote(model_workdir))
                    command = ['ssh',
                               f'{rsync_wrapper_ref.ssh_user}@{rsync_wrapper_ref.ray_head_node}',
                               f'rm -rf {model_workdir}']
                    result = subprocess.run(command)
                    # print(f'delete dir on head node result: {result}')
                rsync_wrapper_ref.upload(model_workdir, if_dir=True)

            return val_score

        except Exception as e:
            print(f'Exception occured: {encoded_solution=} {model_id=} {epoch=}')
            print(traceback.format_exc())
            raise e

    def create_model_and_optimizer(self, encoded_solution, checkpoint_name):
        print('encoded_solution', utils.general.seq_to_str(encoded_solution))
        encoded_solution = [int(x) for x in encoded_solution]
        encoded_solution = tuple(encoded_solution)

        self.hyperparameters.convert_encoding_to_hyperparameters(encoded_solution)

        net = self.function_create_model(hyperparameters=self.hyperparameters,
                                         obs_shape=(9, 84, 84), #hardcoded obs_shape
                                         action_shape=self.action_spec_shape)

        checkpoint = utils.general.try_load_checkpoint(self.folder, checkpoint_name)
        if checkpoint is not None:
            net.load_state_dict(checkpoint['model_state_dict'])

        net_part_names = ['encoder', 'actor', 'critic']
        optimizer = {}
        for k in net_part_names:
            optimizer[k] = self.hyperparameters.get_optimizer(getattr(net, k))

        if checkpoint is not None:
            for k in net_part_names:
                try:
                    optimizer[k].load_state_dict(checkpoint[f'optimizer_state_dict_{k}'])
                except:
                    print(f'Failed to load {k} optimizer state')

                lr, wd = self.hyperparameters.get_optimizer_params()
                print(f'Optimizer (loaded), {k}: {lr=} ; {wd=}')
                optimizer[k] = adjust_optimizer_settings(optimizer[k], lr=lr, wd=wd)

            optimizer_to(optimizer, self.device)

        net = net.to(self.device)

        del checkpoint
        gc.collect()

        return encoded_solution, net, optimizer


    def fitness_no_train(self, encoded_solution, checkpoint_name, eval_env_seed, last_epoch):
        encoded_solution, net, optimizer = self.create_model_and_optimizer(encoded_solution, checkpoint_name)
        # create workdir to write videos to:
        model_workdir = Path(os.path.join(self.folder, f'models/workdir_soup'))
        model_workdir.mkdir(exist_ok=True)
        # common workdir not important: eval doesn't use the replay buffer
        common_workdir = model_workdir

        start_step = last_epoch * self.cfg_rl_general.frames_in_an_epoch // self.cfg_rl_general.action_repeat
        end_step = last_epoch * self.cfg_rl_general.frames_in_an_epoch  # this one will be divided by 2 in the Until in train
        print(f'{start_step=} {end_step=}')

        workspace = Workspace(common_workdir, model_workdir, eval_env_seed, net, optimizer, self.device,
                              self.cfg_rl_general, self.cfg_rl_agent, start_step, end_step, 171717)
        eval_info = workspace.eval()
        return eval_info['episode_reward']