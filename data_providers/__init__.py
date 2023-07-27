from data_providers.cifar import CIFAR10AdvNasDataProvider
from data_providers.stl import STL10AdvNasDataProvider


def get_data_provider(**kwargs):
    if kwargs['dataset_name'] == 'cifar10advnas':
        data_provider = CIFAR10AdvNasDataProvider(**kwargs['dataset_parameters'])
    elif kwargs['dataset_name'] == 'stl10advnas':
        data_provider = STL10AdvNasDataProvider(**kwargs['dataset_parameters'])
    elif kwargs['dataset_name'] == 'mujoco':
        data_provider = None # the issue is that replay buffer is agent-specific
    else:
        raise NotImplementedError

    return data_provider