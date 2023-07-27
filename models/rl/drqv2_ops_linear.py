import torch.nn as nn

from models.rl.spectral_norm_optional import spectral_norm_optional

OPS_linear = {
    "id": lambda *_: Identity(),
    "linear": lambda c_in, c_out, norm, activation_fn, *_: LinearNormAct(c_in, c_out, norm, activation_fn),
    "residual_05": lambda c_in, c_out, norm, activation_fn, *_: LinearResidual(c_in, c_out, norm, activation_fn, 0.5),
    "residual_2": lambda c_in, c_out, norm, activation_fn, *_: LinearResidual(c_in, c_out, norm, activation_fn, 2),
}

class LinearResidual(nn.Module): # from https://arxiv.org/pdf/2106.01151.pdf
    def __init__(self, C_in, C_out, norm, activation_fn, multiplier):
        super(LinearResidual, self).__init__()

        assert norm in ['spectral', None] # layer norm always used in the middle
        assert C_in == C_out

        self.norm_input = nn.LayerNorm(C_in)

        self.linear1 = nn.Linear(C_in, int(C_in * multiplier))
        self.linear1 = spectral_norm_optional(self.linear1, enabled=norm == 'spectral')

        self.linear2 = nn.Linear(int(C_in * multiplier), C_out)
        self.linear2 = spectral_norm_optional(self.linear2, enabled=norm == 'spectral')

        self.act = activation_fn()

    def forward(self, x):
        h = self.norm_input(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.linear2(h)
        return x + h


class LinearNormAct(nn.Module):
    def __init__(self, C_in, C_out, norm, activation_fn):
        super(LinearNormAct, self).__init__()
        norm_layer = None
        if norm == 'layer':
            linear = nn.Linear(C_in, C_out)
            norm_layer = nn.LayerNorm(C_out)
        elif norm in ['spectral', None]:
            linear = spectral_norm_optional(nn.Linear(C_in, C_out), enabled=norm == 'spectral')
        else:
            raise ValueError(norm)

        all = [linear]
        if norm_layer is not None:
            all.append(norm_layer)
        all.append(activation_fn())

        self.op = nn.Sequential(*all)

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x