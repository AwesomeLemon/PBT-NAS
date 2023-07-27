import os
from collections import defaultdict
from copy import deepcopy
from shutil import copyfile

import numpy as np
import torch

import utils


def copy_weights(src_model_id, dst_model_id, folder, epochs, module_dict_keys_to_copy,
                 shrink_perturb, model_for_perturb_state_dict, rl): # for PBT-NAS mixing

    src_file_full_path = os.path.join(folder, 'models', f'model_{src_model_id}_{epochs}')
    dst_file_full_path = os.path.join(folder, 'models', f'model_{dst_model_id}_{epochs}')

    print(f'{src_file_full_path=} {dst_file_full_path=}')
    assert os.path.exists(src_file_full_path)
    assert os.path.exists(dst_file_full_path)

    if module_dict_keys_to_copy is None:
        copyfile(src_file_full_path, dst_file_full_path)
        return

    torch.cuda.empty_cache()
    src_dict = torch.load(src_file_full_path, map_location='cpu')
    dst_dict = torch.load(dst_file_full_path, map_location='cpu')

    src_state_dict = src_dict['model_state_dict']

    def replace(src_model_dict, dst_model_dict, module_dict_keys_to_copy):
        added_param_names = []
        # first need to remove the old values in the dst_model_dict
        # because names of new and old ops can be different
        for i, (k, v) in enumerate(deepcopy(list(dst_model_dict.items()))):
            for k2 in module_dict_keys_to_copy['blocks']:
                if k2 in k:
                    del dst_model_dict[k]

        found = defaultdict(lambda: False)
        ignored_param_names = defaultdict(lambda: [])
        for k, v in src_model_dict.items():
            for k2 in module_dict_keys_to_copy['blocks']:
                if k2 in k:
                    if shrink_perturb is None:
                        dst_model_dict[k] = v
                    else:
                        dst_model_dict[k] = shrink_perturb_tensor(k, v, shrink_perturb,
                                                                  model_for_perturb_state_dict)
                    added_param_names.append(k)
                    found[k2] = True
                else:
                    ignored_param_names[k2].append(k)


    if 'generator' in module_dict_keys_to_copy or 'discriminator' in module_dict_keys_to_copy:
        src_model_state_dict_generator = {k:v for k, v in src_state_dict.items() if 'generator' in k}
        dst_model_state_dict_generator = {k:v for k, v in dst_dict['model_state_dict'].items() if 'generator' in k}
        dst_param_names_before = list(dst_model_state_dict_generator.keys())
        replace(src_model_state_dict_generator, dst_model_state_dict_generator,
                   module_dict_keys_to_copy['generator'])
        dst_dict['model_state_dict'].update(dst_model_state_dict_generator)
        dst_param_names_after = list(dst_model_state_dict_generator.keys())
        for k in dst_param_names_before:
            if k not in dst_param_names_after:
                del dst_dict['model_state_dict'][k]

        # gen_avg
        if_contains_gen_avg = 'avg_gen_state_dict' in src_dict
        if if_contains_gen_avg:
            # backup shrink-perturb params; use None to not shrink-perturb gen_avg; then revert to backup
            sh_pe_backup = deepcopy(shrink_perturb)
            shrink_perturb = None
            avg_gen_src_model_state_dict = deepcopy(src_dict['avg_gen_state_dict'])
            avg_gen_dst_model_state_dict = deepcopy(dst_dict['avg_gen_state_dict'])
            dst_param_names_before = list(avg_gen_dst_model_state_dict.keys())
            replace(avg_gen_src_model_state_dict, avg_gen_dst_model_state_dict,
                       module_dict_keys_to_copy['generator'])
            dst_dict['avg_gen_state_dict'].update(avg_gen_dst_model_state_dict)
            dst_param_names_after = list(avg_gen_dst_model_state_dict.keys())
            for k in dst_param_names_before:
                if k not in dst_param_names_after:
                    del dst_dict['avg_gen_state_dict'][k]
            shrink_perturb = sh_pe_backup

        src_model_state_dict_discriminator = {k:v for k, v in src_state_dict.items() if 'discriminator' in k}
        dst_model_state_dict_discriminator = {k:v for k, v in dst_dict['model_state_dict'].items() if 'discriminator' in k}
        dst_param_names_before = list(dst_model_state_dict_discriminator.keys())
        replace(src_model_state_dict_discriminator, dst_model_state_dict_discriminator,
                   module_dict_keys_to_copy['discriminator'])
        dst_dict['model_state_dict'].update(dst_model_state_dict_discriminator)
        dst_param_names_after = list(dst_model_state_dict_discriminator.keys())
        for k in dst_param_names_before:
            if k not in dst_param_names_after:
                del dst_dict['model_state_dict'][k]

    elif rl:
        def call_replace_for_subset(name, condition_fn):
            src_model_state_dict_subset = {k: v for k, v in src_state_dict.items() if condition_fn(name, k)}
            dst_model_state_dict_subset = {k: v for k, v in dst_dict['model_state_dict'].items() if condition_fn(name, k)}
            # print(f'{src_model_state_dict_subset.keys()=}')
            dst_param_names_before = list(dst_model_state_dict_subset.keys())
            replace(src_model_state_dict_subset, dst_model_state_dict_subset,
                       module_dict_keys_to_copy[name])
            dst_dict['model_state_dict'].update(dst_model_state_dict_subset)
            dst_param_names_after = list(dst_model_state_dict_subset.keys())
            for k in dst_param_names_before:
                if k not in dst_param_names_after:
                    del dst_dict['model_state_dict'][k]

        call_replace_for_subset('actor', lambda name_target, key: name_target in key)
        call_replace_for_subset('critic', lambda name_target, key: name_target in key and 'critic_target' not in key)
        call_replace_for_subset('encoder', lambda name_target, key: name_target in key)

        # critic target - these are averaged weights of the critic => treat the same as avg weight of a GAN
        # note that the names of the exchanged blocks should come from the 'critic', there's nothing separate
        #       for 'critic_target' (since it's just the averaged version of the critic)
        sh_pe_backup = deepcopy(shrink_perturb)
        shrink_perturb = None
        call_replace_for_subset('critic', lambda name_target, key: 'critic_target' in key)
        shrink_perturb = sh_pe_backup

    else:
        raise NotImplementedError()

    print(f"modified weights in model_{dst_model_id}_{epochs} based on model_{src_model_id}_{epochs}")
    torch.save(dst_dict, dst_file_full_path)


def copy_weights_mutate(src_model_id, dst_model_id, folder, epochs, module_dict_keys_to_copy,
                        state_dict_new, rl): # for SEARL-like mutation

    src_file_full_path = os.path.join(folder, 'models', f'model_{src_model_id}_{epochs}')
    dst_file_full_path = os.path.join(folder, 'models', f'model_{dst_model_id}_{epochs}')

    print(f'{src_file_full_path=} {dst_file_full_path=}')
    assert os.path.exists(src_file_full_path)
    assert os.path.exists(dst_file_full_path)

    if module_dict_keys_to_copy is None:
        copyfile(src_file_full_path, dst_file_full_path)
        return

    torch.cuda.empty_cache()
    src_dict = torch.load(src_file_full_path, map_location='cpu')
    dst_dict = torch.load(dst_file_full_path, map_location='cpu')

    src_state_dict = state_dict_new

    def replace(src_model_dict, dst_model_dict, module_dict_keys_to_copy):
        # first need to remove the old values in the dst_model_dict
        # because names of new and old ops can be different
        for i, (k, v) in enumerate(deepcopy(list(dst_model_dict.items()))):
            for k2 in module_dict_keys_to_copy['blocks']:
                if k2 in k:
                    del dst_model_dict[k]

        for k, v in src_model_dict.items():
            for k2 in module_dict_keys_to_copy['blocks']:
                if k2 in k:
                    dst_model_dict[k] = src_model_dict[k]

    if 'generator' in module_dict_keys_to_copy or 'discriminator' in module_dict_keys_to_copy:
        src_model_state_dict_generator = {k:v for k, v in src_state_dict.items() if 'generator' in k}
        dst_model_state_dict_generator = {k:v for k, v in dst_dict['model_state_dict'].items() if 'generator' in k}
        dst_param_names_before = list(dst_model_state_dict_generator.keys())
        replace(src_model_state_dict_generator, dst_model_state_dict_generator,
                   module_dict_keys_to_copy['generator'])
        dst_dict['model_state_dict'].update(dst_model_state_dict_generator)
        dst_param_names_after = list(dst_model_state_dict_generator.keys())
        for k in dst_param_names_before:
            if k not in dst_param_names_after:
                del dst_dict['model_state_dict'][k]

        # gen_avg
        if_contains_gen_avg = 'avg_gen_state_dict' in src_dict
        if if_contains_gen_avg:
            #when mutating, avg_gen should also be changed, use the same values
            avg_gen_src_model_state_dict = deepcopy(src_state_dict)
            # only generator state is needed, plus it shouldn't have the 'generator.' prefix
            avg_gen_src_model_state_dict = {k[len('generator.'):]:v for k, v in avg_gen_src_model_state_dict.items() if 'generator.' in k}
            avg_gen_dst_model_state_dict = deepcopy(dst_dict['avg_gen_state_dict'])
            dst_param_names_before = list(avg_gen_dst_model_state_dict.keys())
            replace(avg_gen_src_model_state_dict, avg_gen_dst_model_state_dict,
                       module_dict_keys_to_copy['generator'])
            dst_dict['avg_gen_state_dict'].update(avg_gen_dst_model_state_dict)
            dst_param_names_after = list(avg_gen_dst_model_state_dict.keys())
            for k in dst_param_names_before:
                if k not in dst_param_names_after:
                    del dst_dict['avg_gen_state_dict'][k]

        src_model_state_dict_discriminator = {k:v for k, v in src_state_dict.items() if 'discriminator' in k}
        dst_model_state_dict_discriminator = {k:v for k, v in dst_dict['model_state_dict'].items() if 'discriminator' in k}
        dst_param_names_before = list(dst_model_state_dict_discriminator.keys())
        replace(src_model_state_dict_discriminator, dst_model_state_dict_discriminator,
                   module_dict_keys_to_copy['discriminator'])
        dst_dict['model_state_dict'].update(dst_model_state_dict_discriminator)
        dst_param_names_after = list(dst_model_state_dict_discriminator.keys())
        for k in dst_param_names_before:
            if k not in dst_param_names_after:
                del dst_dict['model_state_dict'][k]

    elif rl:
        def call_replace_for_subset(name, condition_fn):
            src_model_state_dict_subset = {k: v for k, v in src_state_dict.items() if condition_fn(name, k)}
            dst_model_state_dict_subset = {k: v for k, v in dst_dict['model_state_dict'].items() if condition_fn(name, k)}
            # print(f'{src_model_state_dict_subset.keys()=}')
            dst_param_names_before = list(dst_model_state_dict_subset.keys())
            replace(src_model_state_dict_subset, dst_model_state_dict_subset,
                       module_dict_keys_to_copy[name])
            dst_dict['model_state_dict'].update(dst_model_state_dict_subset)
            dst_param_names_after = list(dst_model_state_dict_subset.keys())
            for k in dst_param_names_before:
                if k not in dst_param_names_after:
                    del dst_dict['model_state_dict'][k]

        call_replace_for_subset('actor', lambda name_target, key: name_target in key)
        call_replace_for_subset('critic', lambda name_target, key: name_target in key and 'critic_target' not in key)
        call_replace_for_subset('encoder', lambda name_target, key: name_target in key)

        # note that the names of the exchanged blocks should come from the 'critic', there's nothing separate
        #       for 'critic_target' (since it's just the averaged version of the critic)
        call_replace_for_subset('critic', lambda name_target, key: 'critic_target' in key)

    else:
        raise NotImplementedError()

    print(f"modified weights in model_{dst_model_id}_{epochs}")
    torch.save(dst_dict, dst_file_full_path)


def genes_to_layer_names_gan(indices_chosen, arch_ss):
    module_dict_keys_to_copy = {'generator': {'blocks': []},
                                'discriminator': {'blocks': []}}

    for ind in indices_chosen:
        position_info = arch_ss.get_pos_of_gene(ind)
        net_name = position_info.network
        block_name = position_info.block
        node_name = position_info.node

        final_name = block_name + '.' + node_name

        module_dict_keys_to_copy[net_name]['blocks'].append(final_name)

    print(f'{indices_chosen=}')
    return module_dict_keys_to_copy


def genes_to_layer_names_rl(indices_chosen, arch_ss):
    module_dict_keys_to_copy = {'actor': {'blocks': []},
                                'encoder': {'blocks': []},
                                'critic': {'blocks': []}}

    for ind in indices_chosen:
        position_info = arch_ss.get_pos_of_gene(ind)
        net_name = position_info.network
        layer_names = position_info.layers

        for layer_name in layer_names:
            final_name = f'.{layer_name}.'
            module_dict_keys_to_copy[net_name]['blocks'].append(final_name)

    print(f'{indices_chosen=}')
    return module_dict_keys_to_copy


def shrink_perturb_tensor(name, tensor, shrink_perturb_pair, model_for_perturb_state_dict):
    if tensor.dtype is torch.long:
        # these all are just num_batches_tracked of the batchnorm
        return tensor

    shrink_coeff, perturb_coeff = shrink_perturb_pair
    reinit = model_for_perturb_state_dict[name]
    # handle the case of mutations where tensor dims don't match
    if np.isclose(shrink_coeff, 0) and np.isclose(perturb_coeff, 1):
        return reinit
    return shrink_coeff * tensor + perturb_coeff * reinit


def shrink_perturb_whole_state(state, shrink_perturb_pair, model_for_perturb_state_dict=None):
    print('shrink_perturb_whole_state')
    new_state = {}
    for k, v in state.items():
        new_state[k] = shrink_perturb_tensor(k, v, shrink_perturb_pair, model_for_perturb_state_dict)
    return new_state
