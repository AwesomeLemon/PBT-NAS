"""
my modification of the torch source code to allow the SpectralNormalized module to be not spectral-normalized
I do this because it makes exchanging weights easier: the models w/o SpectrNorm have the same attributes at the ones with it.
Only forward pass is affected.
"""
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import _SpectralNorm
from typing import Optional

import torch
import torch.nn as nn

class _SpectralNormOptional(_SpectralNorm):
    def __init__(self, weight: torch.Tensor, n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12,
                 enabled=True) -> None:
        super().__init__(weight, n_power_iterations, dim, eps)
        self.enable = enabled

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return weight
        return super(_SpectralNormOptional, self).forward(weight)


def spectral_norm_optional(module: nn.Module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None,
                  enabled=True) -> nn.Module:

    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(module, name, _SpectralNormOptional(weight, n_power_iterations, dim, eps, enabled))
    return module