import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
import utils.rl
from search.rl import dmc
from search.rl.replay_buffer import ReplayBufferStorage, make_replay_loader
from search.rl.replay_buffer_shared_ram import ReplayBufferStorageSharedRam, make_replay_loader_shared_ram
from search.rl.video import VideoRecorder, TrainVideoRecorder


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Workspace:
    def __init__(self, work_dir_common, work_dir_model, seed, agent, optimizer, device, cfg_rl_general, cfg_rl_agent,
                 start_step, end_step, model_id):

        self.device = device
        self.optimizer = optimizer

        self.cfg = cfg_rl_general
        self.cfg_agent = cfg_rl_agent

        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack, self.cfg.action_repeat,
                                  seed, work_dir_model)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack, self.cfg.action_repeat,
                                 seed, work_dir_model)

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        if self.cfg.replay_buffer_type == 'ram':
            self.replay_storage = ReplayBufferStorage(data_specs, work_dir_common)
            fn_make_loader = make_replay_loader
        elif self.cfg.replay_buffer_type == 'ram_shared_memory':
            self.replay_storage = ReplayBufferStorageSharedRam(data_specs, work_dir_common, model_id)
            fn_make_loader = make_replay_loader_shared_ram
        else:
            raise ValueError(self.cfg.replay_buffer_type)

        self.replay_loader = fn_make_loader(work_dir_common, self.cfg.replay_buffer_size,
                                            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
                                            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)

        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            work_dir_model if self.cfg.save_video else None, mode=None)
        self.train_video_recorder = TrainVideoRecorder(
            work_dir_model if self.cfg.save_train_video else None, mode=None)

        self.agent = agent
        self.num_expl_steps = self.cfg_agent.num_expl_steps
        self.stddev_schedule = self.cfg_agent.stddev_schedule
        self.stddev_clip = self.cfg_agent.stddev_clip
        self.critic_target_tau = self.cfg_agent.critic_target_tau
        self.update_every_steps = self.cfg_agent.update_every_steps
        self.aug = RandomShiftsAug(pad=4)

        self.timer = utils.rl.Timer()
        self._global_step = start_step
        self._end_step = end_step

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.rl.Until(self.cfg.num_eval_episodes)

        with torch.no_grad(), utils.rl.eval_mode(self.agent):
            while eval_until_episode(episode):
                time_step = self.eval_env.reset()
                self.video_recorder.init(self.eval_env, enabled=(episode == 0))
                while not time_step.last():
                    action = self.act(time_step.observation, eval_mode=True)
                    time_step = self.eval_env.step(action)
                    self.video_recorder.record(self.eval_env)
                    total_reward += time_step.reward
                    step += 1

                episode += 1
                self.video_recorder.save(f'{self.global_frame}.mp4')

        return {'episode_reward': total_reward / episode,
                'episode_length': step * self.cfg.action_repeat / episode,
                'step': self.global_step}

    def act(self, obs, eval_mode):
        step = self.global_step
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.agent.encoder(obs.unsqueeze(0))
        stddev = utils.rl.schedule(self.stddev_schedule, step)
        dist = self.agent.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def train(self):
        st = time.time()
        # predicates
        train_until_step = utils.rl.Until(self._end_step,
                                          self.cfg.action_repeat)
        seed_until_step = utils.rl.Until(self.cfg.num_seed_frames,
                                         self.cfg.action_repeat)
        eval_every_step = utils.rl.Every(self.cfg.eval_every_frames,
                                         self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation, self.train_env)
        metrics = defaultdict(list)
        if_printed_time_of_first_step = False
        while train_until_step(self.global_step):
            if time_step.last():
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    print(
                        f'step={self.global_step} {episode_reward=:.2f} {total_time=:.1f} fps={episode_frame / elapsed_time:.1f}')

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation, self.train_env)
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(
                    self.global_step + 1):  # added "+1" to evaluate in the end of the interval, not at the beginnin
                st = time.time()
                eval_metrics = self.eval()

                print(f'Eval time: {time.time() - st:.2f}')
                metrics['eval'].append(eval_metrics)

            # sample action
            with torch.no_grad(), utils.rl.eval_mode(self.agent):
                action = self.act(time_step.observation, eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                train_metrics = self.update(if_printed_time_of_first_step)
                metrics['train'].append(train_metrics)

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation, self.train_env)
            episode_step += 1
            self._global_step += 1

            if not if_printed_time_of_first_step:
                print(f'First step time {time.time() - st:.2f}')
                if_printed_time_of_first_step = True

        if self.cfg.replay_buffer_type == 'ram_shared_memory':
            self.replay_loader.dataset.shutdown.value = 1
            for _ in range(10):  # theoretically, just need to do it once, because a batch contains many samples;
                #                 in practice, it fails to clean up everything => 10 batches
                next(self.replay_iter)

        return metrics

    def update(self, if_printed_time_of_first_step):
        replay_iter = self.replay_iter
        step = self.global_step
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        st = time.time()
        batch = next(replay_iter)
        if not if_printed_time_of_first_step:
            print(f'Iter time {time.time() - st:.2f}')
        obs, action, reward, discount, next_obs = utils.rl.to_torch(batch, self.device)

        # augment by random shift
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.agent.encoder(obs)
        with torch.no_grad():
            next_obs = self.agent.encoder(next_obs)

        # update critic
        critic_metrics = self.update_critic(obs, action, reward, discount, next_obs, step)

        metrics['batch_reward'] = reward.mean().item()

        metrics.update(critic_metrics)

        # update actor
        actor_metrics = self.update_actor(obs.detach(), step)
        metrics.update(actor_metrics)

        # update critic target
        utils.rl.soft_update_params(self.agent.critic, self.agent.critic_target,
                                    self.critic_target_tau)

        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.rl.schedule(self.stddev_schedule, step)
            dist = self.agent.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.agent.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.agent.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.optimizer['encoder'].zero_grad(set_to_none=True)
        self.optimizer['critic'].zero_grad(set_to_none=True)
        critic_loss.backward()

        self.optimizer['critic'].step()
        self.optimizer['encoder'].step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.rl.schedule(self.stddev_schedule, step)
        dist = self.agent.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.agent.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.optimizer['actor'].zero_grad(set_to_none=True)
        actor_loss.backward()

        self.optimizer['actor'].step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
