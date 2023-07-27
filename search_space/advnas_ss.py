from typing import NamedTuple, List

from models.gan.advnas_blocks import PRIMITIVES_up, PRIMITIVES, PRIMITIVES_down_my, PRIMITIVES_wo_act


class ArchSearchSpace:
    '''
    The idea is to represent a search space as:
    1) a mapping from int index of a gene to a Position with readable descriptions of what this gene encodes
            This is "gene_to_pos"
    2) a mapping from these descriptions to possible values
            This is "pos_to_options"
    Both of these mappings are populated based on the passed config parameters.
    '''
    def __init__(self):
        pass

class Position(NamedTuple):
    network: str
    block: str
    node: str

class AdvNasMySearchSpace(ArchSearchSpace):
    def __init__(self, model_class, model_parameters, **kwargs):
        super().__init__()
        self.model_class = model_class
        print(f'{model_parameters=}')
        self.model_parameters = model_parameters

        self.n_cells_g = self.model_parameters['n_cells']
        self.len_cell_g = 7
        self.len_generator = self.n_cells_g * self.len_cell_g
        self.if_search_projections = self.model_parameters['if_search_projections']
        if self.if_search_projections:
            self.len_projection = 4
            self.len_projections = (self.len_projection - 1) + self.len_projection * (self.n_cells_g - 1)
            self.len_generator += self.len_projections

        self.n_cells_d = self.model_parameters['n_cells_discr']
        self.advnas_my = self.model_parameters['d_type'] == 'advnas_my'
        if self.model_parameters['d_type'] == 'advnas_my':
            self.len_cell_d = 10
            self.len_discriminator = self.len_cell_d * self.n_cells_d
        elif self.model_parameters['d_type'] == 'autogan':
            self.len_discriminator = 0

        self.len = self.len_generator + self.len_discriminator

        self.gene_to_pos: List = self._create_gene_to_pos()
        self.pos_to_options = self._create_pos_to_options()

    def _create_pos_to_options(self):
        # self.gene_to_pos needs to exist already
        pos_to_options = {}
        for p in self.gene_to_pos:
            if p.network == 'generator':
                if p.block.startswith('cell'):
                    if p.node.startswith('up'):
                        options = PRIMITIVES_up
                    elif p.node.startswith('c'):
                        options = PRIMITIVES
                    else:
                        raise NotImplementedError(p)
                elif p.block.startswith('projections'):
                    if p.node == 'if_enabled_NOWEIGHT':
                        options = [False, True]
                    elif p.node == 'linear':
                        options = [('whole', 'bottom'), ('whole', 'prev'), ('whole', 'target'),
                                   ('part', 'bottom'), ('part', 'prev'), ('part', 'target')]
                    elif p.node == 'up_mode_NOWEIGHT':
                        options = ['nearest', 'bilinear']
                    elif p.node == 'dropout_NOWEIGHT':
                        options = [0.0, 0.1, 0.2]
                    else:
                        raise NotImplementedError(p)
                else:
                    raise NotImplementedError(p)
            else:
                if p.block.startswith('block'):
                    if p.node.startswith('c'):
                        if int(p.block[5:]) == 1 and int(p.node[1:]) <= 2:
                            options = PRIMITIVES_wo_act
                        else:
                            options = PRIMITIVES
                    elif p.node == 'downsample_NOWEIGHT':
                        options = [False, True]
                    elif p.node.startswith('down'):
                        options = PRIMITIVES_down_my
                    else:
                        raise NotImplementedError(p)
                else:
                    raise NotImplementedError(p)

            pos_to_options[p] = options

        return pos_to_options


    def _create_pos_generator(self, ind):
        net_name = 'generator'

        def get_names_from_i(i_cell, i_node):
            cell_name = f'cell{i_cell + 1}'
            if i_node < 2:
                node_name = f'up{i_node}'
            else:
                node_name = f'c{i_node - 2}'
            return cell_name, node_name

        if not self.if_search_projections:
            i_cell = ind // self.len_cell_g
            i_node = ind % self.len_cell_g
            cell_name, node_name = get_names_from_i(i_cell, i_node)
        else:
            if ind < self.len_projections:
                node_names = ['if_enabled_NOWEIGHT', 'linear', 'up_mode_NOWEIGHT', 'dropout_NOWEIGHT']
                if ind < self.len_projection - 1:
                    i_cell = 0
                    i_node = ind
                    node_name = node_names[i_node + 1]  # because no 'if_enabled' for the 0-th cell
                else:
                    i_cell = 1 + (ind - (self.len_projection - 1)) // self.len_projection
                    i_node = (ind - (self.len_projection - 1)) % self.len_projection
                    node_name = node_names[i_node]

                cell_name = f'projections.{i_cell}'
            else:
                ind_shifted = ind - self.len_projections
                i_cell = ind_shifted // self.len_cell_g
                i_node = ind_shifted % self.len_cell_g

                cell_name, node_name = get_names_from_i(i_cell, i_node)

        return Position(net_name, cell_name, node_name)

    def _create_pos_discriminator(self, ind):
        net_name = 'discriminator'

        ind_shifted = ind - self.len_generator

        i_block = ind_shifted // self.len_cell_d
        i_node = ind_shifted % self.len_cell_d

        block_name = f'block{i_block + 1}'
        if i_node < 7:
            node_name = f'c{i_node}'
        elif i_node == 7:  # downsample boolean flag => no weight
            node_name = 'downsample_NOWEIGHT'
        else:
            node_name = f'down{i_node - 8}'

        return Position(net_name, block_name, node_name)

    def _create_gene_to_pos(self):
        gene_to_pos = []
        for ind in range(self.len):
            if ind < self.len_generator:
                pos = self._create_pos_generator(ind)
            else:
                pos = self._create_pos_discriminator(ind)
            print(f'{ind} {pos=}')
            gene_to_pos.append(pos)

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