import numpy as np

from search_space.advnas_ss import ArchSearchSpace
from utils.general import MyAdamW


class Hyperparameters:
    '''
    Contains architecture and optimizer values
    '''
    def __init__(self, arch_ss : ArchSearchSpace, **kwargs):
        self.arch_ss = arch_ss
        self.model_class = kwargs['model_class']
        self.if_gan = self.model_class in ['NeuralFabricGan', 'AdvNas']

        if self.model_class == 'AdvNas':
            self.discr_type = kwargs['model_parameters']['d_type']

        self.optimizer = {'lr': kwargs.get('lr', 0.05), 'weight_decay': kwargs.get('wd', 0.0)}
        self.optimizer.update({'optimizer_name': kwargs['optimizer_name']})

        if self.if_gan:
            self.optimizer['lr_g'] = self.optimizer['lr']
            self.optimizer['lr_d'] = self.optimizer['lr']
            del self.optimizer['lr']

        self.if_search_arch = kwargs.get('if_search_arch', False)
        self.arch = {}
        if self.if_search_arch:
            pass # no need to do anything

        self.alphabet, self.sizes = self.get_search_space_sizes()

    def get_options_architecture(self):
        n_options_per_gene = self.arch_ss.get_n_options_per_gene()
        ops_list = [list(range(n_options_per_gene[i])) for i in range(len(n_options_per_gene))]
        return {'all': ops_list}

    def get_search_space_sizes(self):
        self.sizes = [0]
        self.alphabet = []

        if self.if_search_arch:
            arch_options = self.get_options_architecture()
            for arch_part in arch_options.values():
                for arch_unit in arch_part:
                    self.sizes[0] += 1
                    self.alphabet.append(len(arch_unit))

        print(f'{self.alphabet=}, {self.sizes=}')
        return self.alphabet, self.sizes

    def get_optimizer(self, net, **kwargs):
        if self.if_gan:
            return MyAdamW(net.parameters(), lr=self.optimizer[kwargs['lr_key']],
                               weight_decay=self.optimizer['weight_decay'],
                               betas=(0.0, 0.9), eps=1e-4)
        else:
            return MyAdamW(net.parameters(), lr=self.optimizer['lr'],
                               weight_decay=self.optimizer['weight_decay'])

    def get_optimizer_params(self, **kwargs):
        return self.optimizer[kwargs.get('lr_key', 'lr')], self.optimizer['weight_decay']

    def convert_encoding_to_hyperparameters(self, encoding):
        total_size = np.sum(self.sizes)
        assert len(encoding) == total_size
        encoding = [int(x) for x in encoding]

        encoding_arch, alphabet_arch = encoding[:self.sizes[0]], self.alphabet[:self.sizes[0]]

        if self.if_search_arch:
            self.arch = self.arch_ss.decode(encoding_arch)