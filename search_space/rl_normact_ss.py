from typing import NamedTuple, List, Tuple

from models.rl.drqv2_ops_conv import OPS_conv
from models.rl.drqv2_ops_linear import OPS_linear
from search_space.advnas_ss import ArchSearchSpace


class Position(NamedTuple):
    network: str
    layers: Tuple[str, ...]


class RlNormActSearchSpace(ArchSearchSpace):
    def __init__(self, model_class, model_parameters, **kwargs):
        super().__init__()
        self.model_class = model_class
        self.model_parameters = model_parameters

        self.feature_dims_options = [25, 50, 100, 150]

        self.search_norm_type = model_parameters['search_norm_type']
        assert self.search_norm_type in ['per_network', 'per_layer']
        self.search_activ_type = model_parameters.get('search_activ_type', None)
        assert self.search_activ_type in ['per_network', 'per_layer', None]
        self.activation_options = model_parameters.get('activation_options', None)

        self.if_dont_search_trunc_activ = model_parameters.get('if_dont_search_trunc_activ', False)

        genes_per_layer = 1 + int(self.search_norm_type == 'per_layer') + int(self.search_activ_type == 'per_layer')
        genes_normact_per_layer = int(self.search_norm_type == 'per_layer') + int(self.search_activ_type == 'per_layer')

        self.if_search_actor = model_parameters['search_actor']
        if self.if_search_actor:
            self.n_layers_actor = model_parameters['n_layers_actor'] - 1
            self.len_actor = self.n_layers_actor * genes_per_layer + \
                             genes_normact_per_layer + \
                             1 + int(self.search_activ_type is not None) * int(not self.if_dont_search_trunc_activ) + \
                             int(self.search_norm_type == 'per_network') + \
                             int(self.search_activ_type == 'per_network')
            # layers,norm,activ +
            # maybe norm&activ for 0th layer +
            # feature_dim + maybe trunc_activation +
            # maybe norm for the whole net +
            # maybe activ for the whole net
        else:
            self.len_actor = 0

        self.if_search_critic = model_parameters['search_critic']
        if self.if_search_critic:
            self.n_layers_critic = model_parameters['n_layers_critic'] - 1
            self.len_critic = self.n_layers_critic * genes_per_layer * 2 + \
                              genes_normact_per_layer * 2 + 1 + \
                              int(self.search_activ_type is not None)  * int(not self.if_dont_search_trunc_activ) + \
                              int(self.search_norm_type == 'per_network') * 2 + \
                              int(self.search_activ_type == 'per_network') * 2
            # layers,norm,activ * 2nets + maybe norm&activ for 0th layer * 2nets + feature_dim +
            # maybe trunc_activation + maybe norm for the whole net + maybe activ for the whole net * 2nets
        else:
            self.len_critic = 0

        self.if_search_encoder = model_parameters['search_encoder']
        if self.if_search_encoder:
            self.n_layers_encoder = model_parameters['n_layers_encoder'] - 1

            self.len_encoder = self.n_layers_encoder * genes_per_layer + genes_normact_per_layer + \
                               int(self.search_norm_type == 'per_network') + \
                               int(self.search_activ_type == 'per_network')
            # layers,norm,activ + maybe norm&activ for 0th layer +
            # maybe norm for the whole net +
            # maybe activ for the whole net
        else:
            self.len_encoder = 0

        self.len = self.len_actor + self.len_critic + self.len_encoder

        self.gene_to_pos: List = self._create_gene_to_pos()
        self.pos_to_options = self._create_pos_to_options()

    def _create_pos_to_options(self):
        # self.gene_to_pos needs to exist already
        pos_to_options = {}
        linear_ops = OPS_linear
        for p in self.gene_to_pos:
            if p.network == 'actor':
                if p.layers == ('trunk', 'policy.0'):
                    options = list(self.feature_dims_options)
                elif p.layers[0].startswith('policy'):
                    l_type = p.layers[0].split('.')[0]
                    if l_type == 'policy':
                        options = list(linear_ops.keys())
                    elif l_type == 'policy_norm':
                        options = [None, 'spectral']
                    elif l_type == 'policy_activ':
                        options = list(self.activation_options)
                    else:
                        raise ValueError(p)
                elif p.layers[0] in ['trunc_activ', 'activ']:
                    options = list(self.activation_options)
                elif p.layers == ('norm',):
                    options = [None, 'spectral']
                else:
                    raise ValueError(p)

            elif p.network == 'critic':
                if p.layers == ('trunk', 'Q1.0', 'Q2.0'):
                    options = list(self.feature_dims_options)
                elif p.layers[0].startswith('Q'):
                    l_type_split = p.layers[0].split('.')[0].split('_')
                    if len(l_type_split) == 1: # no '_' => ordinary layer
                        options = list(linear_ops.keys())
                    elif l_type_split[1] == 'norm':
                        options = [None, 'spectral']
                    elif l_type_split[1] == 'activ':
                        options = list(self.activation_options)
                    else:
                        raise ValueError(p)
                elif p.layers[0] in ['trunc_activ', 'activ_Q1', 'activ_Q2']:
                    options = list(self.activation_options)
                elif p.layers[0] in ['norm_Q1', 'norm_Q2']:
                    options = [None, 'spectral']
                else:
                    raise ValueError(p)
                
            elif p.network == 'encoder':
                if p.layers[0].startswith('convnet'):
                    l_type = p.layers[0].split('.')[0]
                    if l_type == 'convnet':
                        options = list(OPS_conv.keys())
                    elif l_type == 'convnet_norm':
                        options = [None, 'spectral']
                    elif l_type == 'convnet_activ':
                        options = list(self.activation_options)
                    else:
                        raise ValueError(p)
                elif p.layers[0] == 'activ':
                    options = list(self.activation_options)
                elif p.layers == ('norm',):
                    options = [None, 'spectral']
                else:
                    raise ValueError(p)
            else:
                raise NotImplementedError(p)

            pos_to_options[p] = options

        return pos_to_options

    def _create_gene_to_pos(self):
        gene_to_pos = []
        def add_pos(net_name, layers):
            pos = Position(net_name, layers)
            print(f'{len(gene_to_pos)} {pos=}')
            gene_to_pos.append(pos)

        # 1) Actor options
        if self.if_search_actor:
            # 1a) ordinary layers
            for ind in range(self.n_layers_actor):
                add_pos('actor', (f'policy.{ind + 1}',))
            # 1b) norms, if needed
            if self.search_norm_type == 'per_layer':
                # for the 0-th layer (the op of which is fixed)
                add_pos('actor', (f'policy_norm.0',))
                # rest
                for ind in range(self.n_layers_actor):
                    add_pos('actor', (f'policy_norm.{ind + 1}',))
            # 1c) activations, if needed
            if self.search_activ_type == 'per_layer':
                # for the 0-th layer (the op of which is fixed)
                add_pos('actor', (f'policy_activ.0',))
                # rest
                for ind in range(self.n_layers_actor):
                    add_pos('actor', (f'policy_activ.{ind + 1}',))
            # 1d) feature_dim
            add_pos('actor', ('trunk', 'policy.0'))
            # 1e) trunc activation, if searching activations
            if (self.search_activ_type is not None) and not self.if_dont_search_trunc_activ:
                add_pos('actor', ('trunc_activ',))
            # 1f) norm per network, if needed
            if self.search_norm_type == 'per_network':
                add_pos('actor', ('norm',))
            # 1g) activ per network, if needed
            if self.search_activ_type == 'per_network':
                add_pos('actor', ('activ',))

        if self.if_search_critic:
            # 2) Critic options (for Q1 and Q2)
            for which_q in [1, 2]:
                for ind in range(self.n_layers_critic):
                    # 2a) ordinary layers
                    add_pos('critic',  (f'Q{which_q}.{ind + 1}',))
                    # 2b) norms, if needed
                    if self.search_norm_type == 'per_layer':
                        add_pos('critic', (f'Q{which_q}_norm.{ind + 1}',))
                    # 2c) activations, if needed
                    if self.search_activ_type == 'per_layer':
                        add_pos('critic', (f'Q{which_q}_activ.{ind + 1}',))
                # 2d) norm for the 0-th layer (the op of which is fixed), if necessary
                if self.search_norm_type == 'per_layer':
                    add_pos('critic', (f'Q{which_q}_norm.0',))
                # 2e) activ for the 0-th layer (the op of which is fixed), if necessary
                if self.search_activ_type == 'per_layer':
                    add_pos('critic', (f'Q{which_q}_activ.0',))
            # 2f) feature_dim
            add_pos('critic', ('trunk', 'Q1.0', 'Q2.0'))
            # 2g) trunc activation, if searching activations
            if (self.search_activ_type is not None) and not self.if_dont_search_trunc_activ:
                add_pos('critic', ('trunc_activ',))
            # 2h) norm per network, if needed
            if self.search_norm_type == 'per_network':
                add_pos('critic', ('norm_Q1',))
                add_pos('critic', ('norm_Q2',))
            # 2i) activ per network, if needed
            if self.search_activ_type == 'per_network':
                add_pos('critic', ('activ_Q1',))
                add_pos('critic', ('activ_Q2',))

        # 3) Encoder options
        if self.if_search_encoder:
            # 3a) ordinary layers
            for ind in range(self.n_layers_encoder):
                add_pos('encoder', (f'convnet.{ind + 1}',))
            # 3b) norms, if needed
            if self.search_norm_type == 'per_layer':
                # for the 0-th layer (the op of which is fixed)
                add_pos('encoder', (f'convnet_norm.0',))
                # rest
                for ind in range(self.n_layers_encoder):
                    add_pos('encoder', (f'convnet_norm.{ind + 1}',))

            # 3c) activations, if needed
            if self.search_activ_type == 'per_layer':
                # for the 0-th layer (the op of which is fixed)
                add_pos('encoder', (f'convnet_activ.0',))
                # rest
                for ind in range(self.n_layers_encoder):
                    add_pos('encoder', (f'convnet_activ.{ind + 1}',))

            # 3d) norm per network, if needed
            if self.search_norm_type == 'per_network':
                add_pos('encoder', ('norm',))

            # 3e) activ per network, if needed
            if self.search_activ_type == 'per_network':
                add_pos('encoder', ('activ',))

        assert len(gene_to_pos) == self.len

        return gene_to_pos

    def get_n_options_per_gene(self):
        return [len(self.pos_to_options[pos]) for pos in self.gene_to_pos]

    def get_pos_of_gene(self, gene):
        if gene < 0 or gene > len(self.gene_to_pos):
            return None
        return self.gene_to_pos[gene]

    def decode(self, genome):
        pos_to_used_op = {}
        for i, g in enumerate(genome):
            pos_cur = self.gene_to_pos[i]
            pos_to_used_op[pos_cur] = self.pos_to_options[pos_cur][g]
        return pos_to_used_op