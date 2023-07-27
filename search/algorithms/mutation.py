'''
copied from https://github.com/automl/SEARL
Implementation of weight mutation
Modified to be fast
Running SEARL is done via ea.py
'''
import numpy as np
import torch

def regularize_weight(weight, mag):
    weight[weight > mag] = mag
    weight[weight < -mag] = -mag
    return weight

def classic_parameter_mutation(state_dict):
    mut_strength = 0.1
    num_mutation_frac = 0.1
    super_mut_strength = 10
    super_mut_prob = 0.05
    reset_prob = super_mut_prob + 0.05

    potential_keys = []
    for i, key in enumerate(state_dict):  # Mutate each param
        if not 'norm' in key:
            W = state_dict[key]
            if len(W.shape) > 1:  # Weights, no bias
                potential_keys.append(key)

    how_many = np.random.randint(1, len(potential_keys) + 1, 1)[0]
    chosen_keys = np.random.choice(potential_keys, how_many, replace=False)

    for key in chosen_keys:
        # References to the variable keys
        W = state_dict[key]
        num_weights = np.prod(W.shape[0])
        # Number of mutation instances
        num_mutations = np.random.randint(int(np.ceil(num_mutation_frac * num_weights)))

        mut_mask = np.random.uniform(0, 1, W.shape) < (num_mutations / num_weights)

        random_num_all = np.random.uniform(0, 1, W.shape)
        mask_supermutation = torch.tensor(mut_mask * (random_num_all < super_mut_prob))
        mask_reset = torch.tensor(mut_mask * ((random_num_all > super_mut_prob) * (random_num_all < reset_prob)))
        mask_normal = torch.tensor(mut_mask * (random_num_all > reset_prob))
        W[mask_supermutation] = W[mask_supermutation] + torch.tensor(np.random.normal(0, np.abs(super_mut_strength * W)).astype('float32'))[mask_supermutation]
        W[mask_reset] = torch.tensor(np.random.normal(0, 1, W.shape).astype('float32'))[mask_reset]
        W[mask_normal] = W[mask_normal] + torch.tensor(np.random.normal(0, np.abs(mut_strength * W)).astype('float32'))[mask_normal]

        state_dict[key] = regularize_weight(W, 1000000)

    # return nothing, because modified inplace


def mutate_weights(ckpt_file_path):
    ckpt_loaded = torch.load(ckpt_file_path, map_location='cpu')
    state_dict = ckpt_loaded['model_state_dict']
    classic_parameter_mutation(state_dict)
    torch.save(ckpt_loaded, ckpt_file_path)
