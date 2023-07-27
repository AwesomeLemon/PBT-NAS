from models.gan.advnas import AdvNasGandD
from models.rl.drqv2 import DrQV2Agent
from functools import partial


def make_function_create_model(**kwargs):
    '''
    partially initializes a model and returns a function that accepts other parameters for the models and creates it
    '''

    if kwargs['model_class'] == 'AdvNas':
        return partial(make_advnas, **kwargs['model_parameters'])
    elif kwargs['model_class'] == 'drqv2':
        return partial(make_rl, **kwargs['model_parameters'])

    raise NotImplementedError

def make_advnas(gf_dim, n_cells, bottom_width, latent_dim, g_activation, df_dim, d_activation, d_spectral_norm, hyperparameters, d_type,
                **kwargs):
    arch = hyperparameters.arch
    model = AdvNasGandD(gf_dim, n_cells, bottom_width, latent_dim, g_activation, arch, df_dim, d_activation, d_spectral_norm, d_type, **kwargs)
    return model

def make_rl(obs_shape, action_shape, feature_dim, hidden_dim, search_encoder, search_actor, search_critic,
            search_q_sep_critic, hyperparameters, task, arch_ss, **kwargs):
    arch = hyperparameters.arch
    model = DrQV2Agent(obs_shape, action_shape, feature_dim, hidden_dim, arch,
                       search_encoder, search_actor, search_critic, search_q_sep_critic, task, arch_ss)
    return model