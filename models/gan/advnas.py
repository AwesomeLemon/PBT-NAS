# modifed code from https://github.com/chengaopro/AdversarialNAS (original header below)
# @Date    : 2019-10-22
# @Author  : Chen Gao

from torch import nn
from models.gan.advnas_blocks import Cell, OptimizedDisBlockAutoGan, DisBlockAutoGan, \
    DisBlockMy, GenProjection, SubsampleLayer, ViewLayer, create_act_advnas


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight.data, 1.)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class AdvNasGandD(nn.Module):
    def __init__(self, gf_dim, n_cells, bottom_width, latent_dim, g_activation, pos_to_used_op,
                 df_dim, d_activation, d_spectral_norm, d_type, **kwargs):
        super().__init__()

        pos_to_used_op_g = {k: v for (k, v) in pos_to_used_op.items() if k.network == 'generator'}
        self.generator = AdvNasGenerator(gf_dim, n_cells, bottom_width, latent_dim, g_activation, pos_to_used_op_g,
                                         kwargs['if_search_projections'], kwargs.get('if_stl_projections', False))

        if d_type == 'advnas_my':
            pos_to_used_op_d = {k: v for (k, v) in pos_to_used_op.items() if k.network == 'discriminator'}
            self.discriminator = AdvNasDiscriminatorMy(df_dim, kwargs['n_cells_discr'], pos_to_used_op_d, d_activation, d_spectral_norm)
        elif d_type == 'autogan':
            self.discriminator = AutoGanDiscriminator(df_dim, d_activation, d_spectral_norm)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

class AdvNasGenerator(nn.Module):
    def __init__(self, gf_dim, n_cells, bottom_width, latent_dim, g_activation, pos_to_used_op,
                 if_search_projections, if_stl_projections, **kwargs):
        super(AdvNasGenerator, self).__init__()
        self.ch = gf_dim
        self.bottom_width = bottom_width
        self.n_cells = n_cells

        if not if_stl_projections:
            if if_search_projections:
                pass
            else:
                fixed_operations = {'linear': ('part', 'target'),
                                    'up_mode_NOWEIGHT': 'nearest',
                                    'dropout_NOWEIGHT': 0.0}
                fixed_operations_full = {'if_enabled_NOWEIGHT': True,
                                         'linear': ('part', 'target'),
                                         'up_mode_NOWEIGHT': 'nearest',
                                         'dropout_NOWEIGHT': 0.0}
                operations_per_proj = [fixed_operations, fixed_operations_full, fixed_operations_full]

            projections = []
            for i in range(n_cells):
                if if_search_projections:
                    ops = {k.node: v for (k, v) in pos_to_used_op.items() if f'projections.{i}' in k.block}
                else:
                    ops = operations_per_proj[i]

                projections.append(GenProjection(i, n_cells, latent_dim, bottom_width, gf_dim, ops))
        else:
            base_latent_dim = latent_dim // 2

            projections = [nn.Sequential(
                                SubsampleLayer(0, base_latent_dim),
                                nn.Linear(base_latent_dim, (self.bottom_width ** 2) * gf_dim),
                                ViewLayer(self.ch, self.bottom_width)),
                           nn.Sequential(
                                SubsampleLayer(base_latent_dim, 2 * base_latent_dim),
                                nn.Linear(base_latent_dim, ((self.bottom_width * 2) ** 2) * gf_dim),
                                ViewLayer(self.ch, self.bottom_width * 2)),
                           None]

        self.projections = nn.ModuleList(projections)

        used_ops_per_cell = []
        for i in range(n_cells):
            ops = {k.node: v for (k, v) in pos_to_used_op.items() if f'cell{i+1}' in k.block}
            used_ops_per_cell.append(ops)

        norm = None
        act = g_activation
        self.cell1 = Cell(gf_dim, gf_dim, 'nearest',  used_ops_per_cell[0], num_skip_in=0, norm=norm, act=act)
        self.cell2 = Cell(gf_dim, gf_dim, 'bilinear', used_ops_per_cell[1], num_skip_in=1, norm=norm, act=act)
        self.cell3 = Cell(gf_dim, gf_dim, 'nearest',  used_ops_per_cell[2], num_skip_in=2, norm=norm, act=act)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(gf_dim), create_act_advnas(act), nn.Conv2d(gf_dim, 3, 3, 1, 1), nn.Tanh()
        )

    def forward(self, z):
        z = z.squeeze()

        projections = [self.projections[i](z) if self.projections[i] is not None else None for i in range(self.n_cells)]

        h1_skip_out, h1 = self.cell1(projections[0])

        if projections[1] is not None:
            h1 = h1 + projections[1]
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out, ))

        if projections[2] is not None:
            h2 = h2 + projections[2]
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))

        output = self.to_rgb(h3)
  
        return output

# AutoGAN-D
class AutoGanDiscriminator(nn.Module):
    def __init__(self, df_dim, activation, d_spectral_norm, **kwargs):
        super(AutoGanDiscriminator, self).__init__()
        self.ch = df_dim
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError
        self.block1 = OptimizedDisBlockAutoGan(3, self.ch, activation, d_spectral_norm)
        self.block2 = DisBlockAutoGan(self.ch, self.ch, activation, d_spectral_norm, downsample=True)
        self.block3 = DisBlockAutoGan(self.ch, self.ch, activation, d_spectral_norm, downsample=False)
        self.block4 = DisBlockAutoGan(self.ch, self.ch, activation, d_spectral_norm, downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)
    
    def forward(self, x):
        h = x
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        try:
            h = model(h)
        except:
            h = model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)
      
        return output


class AdvNasDiscriminatorMy(nn.Module):
    def __init__(self, df_dim, n_cells_discr, pos_to_used_op, d_activation, d_spectral_norm):
        super(AdvNasDiscriminatorMy, self).__init__()
        self.ch = df_dim

        self.activation = create_act_advnas(d_activation)
        # assert len(genotype) % n_cells_discr == 0
        # genotype_per_cell = np.array_split(genotype, n_cells_discr)
        ops_per_cell = []
        for i in range(n_cells_discr):
            ops = {k.node: v for (k, v) in pos_to_used_op.items() if f'block{i + 1}' in k.block}
            ops_per_cell.append(ops)

        self.n_cells = n_cells_discr

        for i in range(self.n_cells):
            block = DisBlockMy(3 if i==0 else self.ch, self.ch, ops_per_cell[i], d_spectral_norm, i == 0, d_activation)
            self.add_module(f'block{i + 1}', block)

        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)

        h = self.activation(h)
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output