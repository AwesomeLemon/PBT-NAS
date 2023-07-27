import torch.nn as nn

from models.rl.spectral_norm_optional import spectral_norm_optional

OPS_conv = {
    "id": lambda *_: Identity(),
    "conv_3": lambda c_in, c_out, stride, norm, activation_fn, *_:
        ConvNormAct(c_in, c_out, 3, stride, 1, norm, activation_fn),
    "conv_5": lambda c_in, c_out, stride, norm, activation_fn, *_:
        ConvNormAct(c_in, c_out, 5, stride, 2, norm, activation_fn),
    "conv_7": lambda c_in, c_out, stride, norm, activation_fn, *_:
        ConvNormAct(c_in, c_out, 7, stride, 3, norm, activation_fn),
    "residual_3": lambda c_in, c_out, stride, norm, activation_fn, *_:
        ConvResidual(c_in, c_out, 3, stride, 1, norm, activation_fn),
    "residual_5": lambda c_in, c_out, stride, norm, activation_fn, *_:
        ConvResidual(c_in, c_out, 5, stride, 2, norm, activation_fn),
    "residual_7": lambda c_in, c_out, stride, norm, activation_fn, *_:
        ConvResidual(c_in, c_out, 7, stride, 3, norm, activation_fn),
    "sep_conv_3": lambda c_in, c_out, stride, norm, activation_fn, *_:
        ConvSep(c_in, c_out, 3, stride, 1, norm, activation_fn),
    "sep_conv_5": lambda c_in, c_out, stride, norm, activation_fn, *_:
        ConvSep(c_in, c_out, 5, stride, 2, norm, activation_fn),
    "sep_conv_7": lambda c_in, c_out, stride, norm, activation_fn, *_:
        ConvSep(c_in, c_out, 7, stride, 3, norm, activation_fn),
}


class ConvResidual(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, norm, activation_fn):
        super(ConvResidual, self).__init__()

        assert norm in ['spectral', None] # layer norm always used in the middle
        assert C_in == C_out

        self.conv1 = nn.Conv2d(C_in, C_out, kernel_size, stride, padding)
        self.conv1 = spectral_norm_optional(self.conv1, enabled=norm == 'spectral')

        self.conv2 = nn.Conv2d(C_out, C_out, kernel_size, stride, padding)
        self.conv2 = spectral_norm_optional(self.conv2, enabled=norm == 'spectral')

        self.act = activation_fn()

    def forward(self, x):
        h = self.conv1(x)
        h = self.act(h)
        h = self.conv2(h)
        h = x + h
        h = self.act(h)
        return h



class ConvNormAct(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, norm, activation_fn):
        super(ConvNormAct, self).__init__()
        norm_layer = None
        if norm == 'layer':
            conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding)
            norm_layer = nn.LayerNorm(C_out)
        elif norm in ['spectral', None]:
            conv = spectral_norm_optional(nn.Conv2d(C_in, C_out, kernel_size, stride, padding), enabled=norm == 'spectral')
        else:
            raise ValueError(norm)

        all = [conv]
        if norm_layer is not None:
            all.append(norm_layer)
        all.append(activation_fn())

        self.op = nn.Sequential(*all)

    def forward(self, x):
        return self.op(x)


class ConvSep(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, norm, activation_fn):
        super(ConvSep, self).__init__()

        assert norm in ['spectral', None] # layer norm always used in the middle
        assert C_in == C_out

        self.conv1 = nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in)
        self.conv1 = spectral_norm_optional(self.conv1, enabled=norm == 'spectral')

        self.conv2 = nn.Conv2d(C_in, C_out, 1, stride, 0)
        self.conv2 = spectral_norm_optional(self.conv2, enabled=norm == 'spectral')

        self.act = activation_fn()

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.act(h)
        return h

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x