from search_space.advnas_ss import AdvNasMySearchSpace
from search_space.rl_normact_ss import RlNormActSearchSpace


def get_arch_search_space(**kwargs):
    if kwargs['model_class'] == 'AdvNas':
        arch_ss = AdvNasMySearchSpace(**kwargs)
    elif kwargs['model_class'] == 'drqv2':
        if kwargs['model_parameters']['task'] == 'rl_normact':
            arch_ss = RlNormActSearchSpace(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return arch_ss