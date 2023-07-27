# based on https://github.com/facebookresearch/drqv2/blob/main/drqv2.py , the original header is below
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from models.rl.drqv2_nets_normact import CriticNormActNAS, ActorNormActNAS, EncoderNormActNAS
from models.rl.drqv2_nets_original import Encoder, Actor, Critic

class DrQV2Agent(nn.Module):
    def __init__(self, obs_shape, action_shape, feature_dim, hidden_dim, pos_to_used_op,
                 search_encoder, search_actor, search_critic, search_q_sep_critic, task, arch_ss):
        super().__init__()

        if not search_encoder:
            self.encoder = Encoder(obs_shape)
        else:
            pos_to_used_op_encoder = {k: v for (k, v) in pos_to_used_op.items() if k.network == 'encoder'}
            self.encoder = EncoderNormActNAS(obs_shape, pos_to_used_op_encoder, arch_ss)

        if not search_actor:
            self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim)
        else:
            pos_to_used_op_actor = {k: v for (k, v) in pos_to_used_op.items() if k.network == 'actor'}
            actor_feature_dim_maybe = [v for (k, v) in pos_to_used_op_actor.items() if k.layers == ('trunk', 'policy.0')]
            if len(actor_feature_dim_maybe) > 0:
                assert len(actor_feature_dim_maybe) == 1
                feature_dim_actor = actor_feature_dim_maybe[0]
            else:
                feature_dim_actor = feature_dim
            self.actor = ActorNormActNAS(self.encoder.repr_dim, action_shape, feature_dim_actor, hidden_dim,
                                         pos_to_used_op_actor, arch_ss)

        if not search_critic:
            self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim)
            self.critic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim)
        else:
            pos_to_used_op_critic = {k: v for (k, v) in pos_to_used_op.items() if k.network == 'critic'}
            critic_feature_dim_maybe = [v for (k, v) in pos_to_used_op_critic.items() if k.layers == ('trunk', 'Q1.0', 'Q2.0')]
            if len(critic_feature_dim_maybe) > 0:
                assert len(critic_feature_dim_maybe) == 1
                feature_dim_critic = critic_feature_dim_maybe[0]
            else:
                feature_dim_critic = feature_dim

            self.critic = CriticNormActNAS(self.encoder.repr_dim, action_shape, feature_dim_critic, hidden_dim,
                                           pos_to_used_op_critic, search_q_sep_critic, arch_ss)
            self.critic_target = CriticNormActNAS(self.encoder.repr_dim, action_shape, feature_dim_critic, hidden_dim,
                                                  pos_to_used_op_critic, search_q_sep_critic, arch_ss)

        self.critic_target.load_state_dict(self.critic.state_dict()) # I leave this for the sake of the 0-th iteration;
        #                          afterwards, this is unnecessary because I load critic_target state in fitness_fun_rl

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)