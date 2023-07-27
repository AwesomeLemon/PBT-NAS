# modifed code from https://github.com/chengaopro/AdversarialNAS (original header below)
# @Date    : 2019-10-22
# @Author  : Chen Gao
import torch
from torch import nn
import torch.nn.functional as F

# 7
PRIMITIVES = [
    'none',
    'skip_connect',
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# 5
PRIMITIVES_wo_act = [
  'conv_1x1',
  'conv_3x3',
  'conv_5x5',
  'dil_conv_3x3',
  'dil_conv_5x5'
]

# 3
PRIMITIVES_up = [
    'nearest',
    'bilinear',
    'ConvTranspose'
]

# 6
PRIMITIVES_down = [
    'avg_pool',
    'max_pool',
    'conv_3x3',
    'conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# 8
PRIMITIVES_down_my = [
    'avg_pool_input',
    'max_pool_input',
    'avg_pool',
    'max_pool',
    'conv_3x3',
    'conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# ------------------------------------------------------------------------------------------------------------------- #

OPS = {
    'none': lambda in_ch, out_ch, stride, sn, act: Zero(),
    'skip_connect': lambda in_ch, out_ch, stride, sn, act: Identity(),
    'conv_1x1': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 1, stride, 0, sn, act),
    'conv_3x3': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 3, stride, 1, sn, act),
    'conv_5x5': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 5, stride, 2, sn, act),
    'dil_conv_3x3': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 3, stride, 2, 2, sn, act),
    'dil_conv_5x5': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 5, stride, 4, 2, sn, act)
}

OPS_down = {
    'avg_pool': lambda in_ch, out_ch, stride, sn, act: Pool(in_ch, out_ch, mode='Avg'),
    'max_pool': lambda in_ch, out_ch, stride, sn, act: Pool(in_ch, out_ch, mode='Max'),
    'conv_3x3': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 3, stride, 1, sn, act),
    'conv_5x5': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 5, stride, 2, sn, act),
    'dil_conv_3x3': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 3, stride, 2, 2, sn, act),
    'dil_conv_5x5': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 5, stride, 4, 2, sn, act)
}

OPS_down_my = {
    'avg_pool_input': lambda in_ch, out_ch, stride, sn, act: Pool(in_ch, out_ch, mode='Avg'), # exactly the same as avg_pool
    'max_pool_input': lambda in_ch, out_ch, stride, sn, act: Pool(in_ch, out_ch, mode='Max'), # exactly the same as max_pool
    'avg_pool': lambda in_ch, out_ch, stride, sn, act: Pool(in_ch, out_ch, mode='Avg'),
    'max_pool': lambda in_ch, out_ch, stride, sn, act: Pool(in_ch, out_ch, mode='Max'),
    'conv_3x3': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 3, stride, 1, sn, act),
    'conv_5x5': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 5, stride, 2, sn, act),
    'dil_conv_3x3': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 3, stride, 2, 2, sn, act),
    'dil_conv_5x5': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 5, stride, 4, 2, sn, act)
}

UPS = {
    'nearest': lambda in_ch, out_ch, act: Up(in_ch, out_ch, act, mode='nearest'),
    'bilinear': lambda in_ch, out_ch, act: Up(in_ch, out_ch, act, mode='bilinear'),
    'ConvTranspose': lambda in_ch, out_ch, act: Up(in_ch, out_ch, act, mode='convT')
}


# ------------------------------------------------------------------------------------------------------------------- #

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, sn, act):
        super(Conv, self).__init__()
        if sn:
            conv = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding))
        else:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        if act is not None:
            self.op = nn.Sequential(create_act_advnas(act), conv)
        else:
            self.op = nn.Sequential(conv)

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, sn, act):
        super(DilConv, self).__init__()
        if sn:
            dilconv = nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
        else:
            dilconv = \
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        if act is not None:
            self.op = nn.Sequential(create_act_advnas(act), dilconv)
        else:
            self.op = nn.Sequential(dilconv)

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, act, mode=None):
        super(Up, self).__init__()
        self.up_mode = mode
        if self.up_mode == 'convT':
            self.convT = nn.Sequential(
                create_act_advnas(act),
                nn.ConvTranspose2d(
                    in_ch, in_ch, kernel_size=3, stride=2, padding=1, output_padding=1, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)
            )
        else:
            self.c = nn.Sequential(
                create_act_advnas(act),
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
            )

    def forward(self, x):
        if self.up_mode == 'convT':
            return self.convT(x)
        else:
            return self.c(F.interpolate(x, scale_factor=2, mode=self.up_mode, align_corners=False if self.up_mode == 'bilinear' else None))


class Pool(nn.Module):
    def __init__(self, in_ch, out_ch, mode=None):
        super(Pool, self).__init__()
        if mode == 'Avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        elif mode == 'Max':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

    def forward(self, x):
        return self.pool(x)


class MixedOp(nn.Module):
    def __init__(self, in_ch, out_ch, stride, sn, act, primitives):
        super(MixedOp, self).__init__()
        self.ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](in_ch, out_ch, stride, sn, act)
            self.ops.append(op)

    def forward(self, x):
        return sum(op(x) for op in self.ops)


class MixedUp(nn.Module):
    def __init__(self, in_ch, out_ch, act, primitives):
        super(MixedUp, self).__init__()
        self.ups = nn.ModuleList()
        for primitive in primitives:
            up = UPS[primitive](in_ch, out_ch, act)
            self.ups.append(up)

    def forward(self, x):
        return sum(up(x) for up in self.ups)


class MixedOpMy(nn.Module): # can pass any OPS list
    def __init__(self, in_ch, out_ch, stride, sn, act, primitives, ops_list):
        super(MixedOpMy, self).__init__()
        self.ops = nn.ModuleList()
        for primitive in primitives:
            op = ops_list[primitive](in_ch, out_ch, stride, sn, act)
            self.ops.append(op)

    def forward(self, x):
        return sum(op(x) for op in self.ops)


# ------------------------------------------------------------------------------------------------------------------- #


class Cell(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode, used_ops_dict, num_skip_in=0, norm=None, act='relu'):
        super(Cell, self).__init__()

        self.up0 = MixedUp(in_channels, out_channels, act, [used_ops_dict['up0']])
        self.up1 = MixedUp(in_channels, out_channels, act, [used_ops_dict['up1']])
        if used_ops_dict['c0'] != 'none':
            self.c0 = MixedOp(out_channels, out_channels, 1, False, act, [used_ops_dict['c0']])
        if used_ops_dict['c1'] != 'none':
            self.c1 = MixedOp(out_channels, out_channels, 1, False, act, [used_ops_dict['c1']])
        if used_ops_dict['c2'] != 'none':
            self.c2 = MixedOp(out_channels, out_channels, 1, False, act, [used_ops_dict['c2']])
        if used_ops_dict['c3'] != 'none':
            self.c3 = MixedOp(out_channels, out_channels, 1, False, act, [used_ops_dict['c3']])
        if used_ops_dict['c4'] != 'none':
            self.c4 = MixedOp(out_channels, out_channels, 1, False, act, [used_ops_dict['c4']])

        self.up_mode = up_mode
        self.norm = norm

        # cross scale skip
        self.skip_in_ops = None
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_skip_in)]
            )

    def forward(self, x, skip_ft=None):

        node0 = self.up0(x)
        node1 = self.up1(x)
        _, _, ht, wt = node0.size()

        # for different topologies
        if hasattr(self, 'c0'):
            node2 = self.c0(node0)
            if hasattr(self, 'c1'):
                node2 = node2 + self.c1(node1)
        else:
            # modification to make not-nice genotypes still produce viable architectures:
            if hasattr(self, 'c1'):
                node2 = self.c1(node1)
            else:
                node2 = node0 + node1

        # skip out feat
        h_skip_out = node2

        # skip in feat
        if self.skip_in_ops:
            assert len(self.skip_in_ops) == len(skip_ft)
            for ft, skip_in_op in zip(skip_ft, self.skip_in_ops):
                node2 += skip_in_op(F.interpolate(ft, size=(ht, wt), mode=self.up_mode, align_corners=False if self.up_mode == 'bilinear' else None))

        # for different topologies
        if hasattr(self, 'c2'):
            node3 = self.c2(node0)
            if hasattr(self, 'c3'):
                node3 = node3 + self.c3(node1)
                if hasattr(self, 'c4'):
                    node3 = node3 + self.c4(node2)
            else:
                if hasattr(self, 'c4'):
                    node3 = node3 + self.c4(node2)
        else:
            if hasattr(self, 'c3'):
                node3 = self.c3(node1)
                if hasattr(self, 'c4'):
                    node3 = node3 + self.c4(node2)
            else:
                # my modification to make not-nice genotypes still produce viable architectures:
                if hasattr(self, 'c4'):
                    node3 = self.c4(node2)
                else:
                    node3 = node2

        return h_skip_out, node3


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)

class OptimizedDisBlockAutoGan(nn.Module):
    def __init__(self, in_channels, out_channels, activation, d_spectral_norm, ksize=3, pad=1):
        super(OptimizedDisBlockAutoGan, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlockAutoGan(nn.Module):
    def __init__(self, in_channels, out_channels, activation, d_spectral_norm, hidden_channels=None, downsample=False):
        super(DisBlockAutoGan, self).__init__()
        self.ksize = 3
        self.pad = 1
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=self.ksize, padding=self.pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=self.ksize, padding=self.pad)
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlockMy(nn.Module):
    def __init__(self, in_channels, out_channels, used_ops_dict, d_spectral_norm, is_first_block, act):
        super(DisBlockMy, self).__init__()
        for i in range(5 + 2):
            if_operate_on_image = is_first_block and i <= 2

            if used_ops_dict[f'c{i}'] == 'none' and not if_operate_on_image: # because if if_operate_on_image, "0" is not a noop
                continue

            in_channels_to_use = in_channels
            if is_first_block and not (i <= 2):
                in_channels_to_use = out_channels

            op = MixedOpMy(in_channels_to_use, out_channels, 1,
                         d_spectral_norm, None if if_operate_on_image else act,
                         [used_ops_dict[f'c{i}']], OPS)

            self.add_module(f'c{i}', op)

        self.downsample = used_ops_dict['downsample_NOWEIGHT'] # 0/1

        self.down0 = MixedOpMy(out_channels, out_channels, 2, d_spectral_norm, act, [used_ops_dict['down0']], OPS_down_my)
        self.down1 = MixedOpMy(out_channels, out_channels, 2, d_spectral_norm, act, [used_ops_dict['down1']], OPS_down_my)

        self.if_down0_on_input = used_ops_dict['down0'] in ['avg_pool_input', 'max_pool_input']
        self.if_down1_on_input = used_ops_dict['down1'] in ['avg_pool_input', 'max_pool_input']

    def forward(self, x):
        op_present = [hasattr(self, f'c{i}') for i in range(4+2 + 1)]
        node_values = [None for _ in range(3+2 + 1)] # there are fewer nodes than ops
        if sum(op_present[:3]) == 0: # no ops immediately after input => return input
            return x

        for i in range(3):
            if op_present[i]:
                node_values[i] = getattr(self, f'c{i}')(x)

        if node_values[1] is not None:
            if self.downsample and self.if_down0_on_input:
                node_values[1] = self.down0(node_values[1])

            if op_present[3] and node_values[0] is not None:
                value_to_add = self.c3(node_values[0])
                if self.downsample and self.if_down0_on_input:
                    value_to_add = self.down0(value_to_add)

                node_values[1] = node_values[1] + value_to_add

            # my search space: an additional op
            if op_present[5]:
                node_values[1] = self.c5(node_values[1])

        if node_values[2] is not None:
            if self.downsample and self.if_down1_on_input:
                node_values[2] = self.down1(node_values[2])

            if op_present[4] and node_values[0] is not None:
                value_to_add = self.c4(node_values[0])

                if self.downsample and self.if_down1_on_input:
                    value_to_add = self.down1(value_to_add)

                node_values[2] = node_values[2] + value_to_add

            # my search space: an additional op
            if op_present[6]:
                node_values[2] = self.c6(node_values[2])

        if node_values[1] is None and node_values[2] is None:
            assert node_values[0] is not None, 'Impossible: I checked that at least 1 of 3 ops was present'
            return node_values[0]

        if node_values[1] is not None:
            if self.downsample and not self.if_down0_on_input:
                node_values[1] = self.down0(node_values[1])

            if node_values[2] is not None:
                if self.downsample and not self.if_down1_on_input:
                    node_values[2] = self.down1(node_values[2])

                node_values[3] = node_values[1] + node_values[2]

            node_values[3] = node_values[1]
        elif node_values[2] is not None:
            if self.downsample and not self.if_down1_on_input:
                node_values[2] = self.down1(node_values[2])

            node_values[3] = node_values[2]
        else:
            raise ValueError('by this point one of the nodes should not be None')

        return node_values[3]


class GenProjection(nn.Module):
    def __init__(self, i_projection, n_projections, latent_dim, bottom_width, gf_dim, used_ops_dict):
        '''
        Genome for 3 projections from noise: l1, l2, l3
        The search space is:
            - enabled no/yes - always yes for l1
            - (noise_dim[whole or 1/3] x target_dim[bottom_width, prev_width, target_width])
            - upsample /*if needed*/: nearest, bilinear
            -  dropout (0.0, 0.1, 0.2)
        '''
        super(GenProjection, self).__init__()
        self.gf_dim = gf_dim

        self.if_enabled = True if i_projection == 0 else used_ops_dict['if_enabled_NOWEIGHT']

        in_option, out_option = used_ops_dict['linear']
        in_dim = latent_dim if in_option == 'whole' else latent_dim // n_projections
        self.in_subsample = lambda h: h if in_option == 'whole' else h[:, i_projection * in_dim:(i_projection + 1) * in_dim]

        self.target_width = bottom_width * 2 ** i_projection
        out_width = {'bottom': bottom_width,
                     'prev': int(bottom_width * 2 ** (i_projection - 1)),
                     'target': self.target_width}[out_option]
        self.out_width = out_width
        out_dim = (out_width ** 2) * gf_dim
        self.linear = nn.Linear(in_dim, out_dim)

        self.scale_factor = self.target_width // out_width
        self.up_mode = used_ops_dict['up_mode_NOWEIGHT'] #only parameterless ops because should work for any scale factor

        self.dropout = nn.Dropout(used_ops_dict['dropout_NOWEIGHT'])

    def forward(self, x):
        if not self.if_enabled:
            return None
        x = self.in_subsample(x)
        x = self.linear(x)
        x = x.view(-1, self.gf_dim, self.out_width, self.out_width)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.up_mode, align_corners=False if self.up_mode == 'bilinear' else None)
        x = self.dropout(x)
        return x


class SubsampleLayer(torch.nn.Module):
    def __init__(self, range_start, range_end):
        super().__init__()
        self.range_start = range_start
        self.range_end = range_end

    def forward(self, x):
        return x[:, self.range_start:self.range_end]


class ViewLayer(torch.nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.channels = channels
        self.width = width

    def forward(self, x):
        return x.view(-1, self.channels, self.width, self.width)


def create_act_advnas(act_name):
    if act_name == 'relu':
        return torch.nn.ReLU()
    elif act_name == 'lrelu001':
        return torch.nn.LeakyReLU()
    elif act_name == 'swish':
        return torch.nn.SiLU()
    else:
        raise NotImplementedError(act_name)
