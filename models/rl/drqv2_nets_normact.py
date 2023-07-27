import torch
from torch import nn as nn

import utils
import utils.rl
from models.rl.drqv2_ops_linear import LinearNormAct, OPS_linear
from models.rl.drqv2_ops_conv import ConvNormAct, OPS_conv

activ_name_to_class = {'id': nn.Identity, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'swish': nn.SiLU}

class EncoderNormActNAS(nn.Module):
    def __init__(self, obs_shape, pos_to_used_op_encoder, arch_ss):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 41 * 41

        ops = {k.layers[0]: v for (k, v) in pos_to_used_op_encoder.items() if len(k.layers) == 1}

        norm = ops.get('norm', None)  # disabled by default

        def get_norm(i_op):
            if arch_ss.search_norm_type == 'per_network':
                return norm
            return ops[f'convnet_norm.{i_op}']

        def get_activ(i_op):
            if arch_ss.search_activ_type is None:
                return nn.ReLU
            if arch_ss.search_activ_type == 'per_network':
                return activ_name_to_class[ops[f'activ']]
            return activ_name_to_class[ops[f'convnet_activ.{i_op}']]

        convnet = [ConvNormAct(obs_shape[0], 32, 3, 2, 0, get_norm(0), get_activ(0))]
        for i_op in range(1, arch_ss.n_layers_encoder + 1):
            op = OPS_conv[ops[f'convnet.{i_op}']](32, 32, 1, get_norm(i_op), get_activ(i_op))
            convnet.append(op)

        self.convnet = nn.Sequential(*convnet)

        self.apply(utils.rl.weight_init_rl)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)

        return h


class ViewFlatten(nn.Module):
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


def create_trunk_modules(arch_ss, feature_dim, ops, repr_dim):
    trunk_activ = nn.Tanh if (arch_ss.search_activ_type is None) or (arch_ss.if_dont_search_trunc_activ) else activ_name_to_class[ops['trunc_activ']]
    trunk_modules = nn.Sequential(ViewFlatten(), LinearNormAct(repr_dim, feature_dim, 'layer', trunk_activ))
    return trunk_modules, feature_dim

class ActorNormActNAS(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, pos_to_used_op_actor,
                 arch_ss):
        super().__init__()
        linear_ops = OPS_linear

        ops = {k.layers[0]: v for (k, v) in pos_to_used_op_actor.items() if len(k.layers) == 1}
        # print(f'Actor ops: {ops}')

        trunk_modules, feature_dim = create_trunk_modules(arch_ss, feature_dim, ops, repr_dim)
        self.trunk = nn.Sequential(*trunk_modules)

        norm = ops.get('norm', None)  # disabled by default

        def get_norm(i_op):
            if arch_ss.search_norm_type == 'per_network':
                return norm
            return ops[f'policy_norm.{i_op}']

        def get_activ(i_op):
            if arch_ss.search_activ_type is None:
                return nn.ReLU
            if arch_ss.search_activ_type == 'per_network':
                return activ_name_to_class[ops[f'activ']]
            return activ_name_to_class[ops[f'policy_activ.{i_op}']]

        policy = [LinearNormAct(feature_dim, hidden_dim, get_norm(0), get_activ(0))]
        for i_op in range(1, arch_ss.n_layers_actor + 1):
            # always use spectral norm
            op = linear_ops[ops[f'policy.{i_op}']](hidden_dim, hidden_dim, get_norm(i_op), get_activ(i_op))
            policy.append(op)
        self.policy = nn.Sequential(*policy)

        self.head = nn.Linear(hidden_dim, action_shape[0])  # sn-rl paper says that spectral norm isn't applied to 1st & last layers

        self.apply(utils.rl.weight_init_rl)

        # print('Printing actor: ', self)

    def forward(self, obs, std):
        h = self.trunk(obs)
        # print(f'{h[0][0]=}')
        h = self.policy(h)
        mu = self.head(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.rl.TruncatedNormal(mu, std)
        return dist


class CriticNormActNAS(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, pos_to_used_op_actor, search_q_sep_critic,
                 arch_ss):
        super().__init__()

        assert search_q_sep_critic
        ops = {k.layers[0]: v for (k, v) in pos_to_used_op_actor.items() if
               len(k.layers) == 1}  # contains ops for both Q1 & Q2

        # print(f'Critic ops: {ops}')

        trunk_modules, feature_dim = create_trunk_modules(arch_ss, feature_dim, ops, repr_dim)
        self.trunk = nn.Sequential(*trunk_modules)

        norm_Q1 = ops.get('norm_Q1', None)  # disabled by default
        norm_Q2 = ops.get('norm_Q2', None)  # disabled by default

        def get_norm(which_q, i_op):
            if arch_ss.search_norm_type == 'per_network':
                return {1: norm_Q1, 2: norm_Q2}[which_q]
            return ops[f'Q{which_q}_norm.{i_op}']

        def get_activ(which_q, i_op):
            if arch_ss.search_activ_type is None:
                return nn.ReLU
            if arch_ss.search_activ_type == 'per_network':
                return activ_name_to_class[ops[f'activ_Q{which_q}']]
            return activ_name_to_class[ops[f'Q{which_q}_activ.{i_op}']]

        Q1 = [LinearNormAct(feature_dim + action_shape[0], hidden_dim, get_norm(1, 0), get_activ(1, 0))]

        linear_ops = OPS_linear
        for i_op in range(1, arch_ss.n_layers_critic + 1):
            op = linear_ops[ops[f'Q1.{i_op}']](hidden_dim, hidden_dim, get_norm(1, i_op), get_activ(1, i_op))
            Q1.append(op)
        self.Q1 = nn.Sequential(*Q1)

        self.Q1_head = nn.Linear(hidden_dim, 1)  # sn-rl paper says that spectral norm isn't applied to 1st & last layers

        Q2 = [LinearNormAct(feature_dim + action_shape[0], hidden_dim, get_norm(2, 0), get_activ(2, 0))]

        for i_op in range(1, arch_ss.n_layers_critic + 1):
            op = linear_ops[ops[f'Q2.{i_op}']](hidden_dim, hidden_dim, get_norm(2, i_op), get_activ(2, i_op))
            Q2.append(op)
        self.Q2 = nn.Sequential(*Q2)
        self.Q2_head = nn.Linear(hidden_dim, 1)  # sn-rl paper says that spectral norm isn't applied to 1st & last layers

        self.apply(utils.rl.weight_init_rl)

        # print('Printing critic: ', self)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)

        q1 = self.Q1(h_action)
        q1 = self.Q1_head(q1)

        q2 = self.Q2(h_action)
        q2 = self.Q2_head(q2)

        return q1, q2
