from pathlib import Path

import os

import gc

import time
import torch.nn

import tabulate
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F

import utils
from cleanfid import fid

import utils.general


def _train_epoch_gan(device, loader, model, criterion, optimizer, **kwargs):
    generator_batch_size_multiplier = kwargs.get('generator_batch_size_multiplier', 4)
    print(f'generator batch size is {generator_batch_size_multiplier} * batch_size')

    loss_sum = 0.0
    d_loss_sum = 0.0
    n_samples = 0

    criterion.to(device)
    if_weight_avg = 'gen_avg_params' in kwargs

    hyperparameters = kwargs['hyperparameters']

    # print('d_to_g_train_ratio = 5 by default')
    d_to_g_train_ratio = hyperparameters.optimizer.get('d_to_g_train_ratio', 5)
    d_train_freq, g_train_freq = 1, 1
    if d_to_g_train_ratio < 1:
        d_train_freq = int(1 / d_to_g_train_ratio)
    else:
        g_train_freq = int(d_to_g_train_ratio)

    model.generator.train()
    model.discriminator.train()

    optimizer_g = optimizer['generator']
    optimizer_d = optimizer['discriminator']

    lr_g = optimizer_g.defaults['lr']
    for param_group in optimizer_g.param_groups:
        param_group['lr'] = lr_g

    lr_d = optimizer_d.defaults['lr']
    for param_group in optimizer_d.param_groups:
        param_group['lr'] = lr_d

    latent_size = kwargs.get('latent_dim', 100)

    for i, (input, _) in enumerate(loader):
        input = input.to(device)
        len_input = input.shape[0]

        if i % d_train_freq == 0:  # train discriminator
            optimizer_d.zero_grad()

            output_d_real = model.discriminator(input)
            real_loss = torch.mean(F.relu(1.0 - output_d_real))

            z = torch.randn(len_input, latent_size, device=device)
            z = z.reshape(*z.shape, 1, 1)  # (len_input, latent_dim, 1, 1)

            gen_imgs = model.generator(z)
            output_d_fake = model.discriminator(gen_imgs.detach())
            fake_loss = torch.mean(F.relu(1.0 + output_d_fake))

            d_loss = fake_loss + real_loss
            d_loss.backward()

            optimizer_d.step()
            d_loss_sum += (0.5 * fake_loss.item() + 0.5 * real_loss.item()) * len_input

        if i % g_train_freq == 0:  # train generator
            optimizer_g.zero_grad()

            z = torch.randn(len_input * generator_batch_size_multiplier, latent_size, device=device)
            z = z.reshape(*z.shape, 1, 1)  # (len_input, latent_dim, 1, 1)
            gen_imgs = model.generator(z)

            output_d_fake2 = model.discriminator(gen_imgs)
            generator_loss = -torch.mean(output_d_fake2)

            generator_loss.backward()

            optimizer_g.step()
            loss_sum += generator_loss.item() * len_input

            if if_weight_avg:
                # moving average weight
                for p, avg_p in zip(model.generator.parameters(), kwargs['gen_avg_params']):
                    avg_p.mul_(0.999).add_(p.data, alpha=0.001)

        n_samples += len_input

    to_return = {'loss': loss_sum / n_samples, 'd_loss': d_loss_sum / n_samples}

    return to_return

def _evaluate_gan(device, model, **kwargs):
    torch.cuda.empty_cache()
    model.generator.eval()
    model.discriminator.eval()

    latent_size = kwargs.get('latent_dim', 100)
    print(f'{kwargs["fid_dataset"]=} {kwargs["fid_split"]=}')

    gen = lambda z: model.generator(z.reshape(*z.shape, 1, 1)).mul_(127.5).add_(127.5).clamp_(0.0, 255.0)
    # determine resolution:
    with torch.no_grad(), torch.cuda.amp.autocast():
        z = torch.randn(2, latent_size,
                        device=device)  # wanna have 1, not 2, but squeeze in advnas removes the dim then.
        resolution = gen(z).shape[-1]
        print(f'{resolution=}')

    fid_score = fid.compute_fid(gen=gen, dataset_name=kwargs.get('fid_dataset', "cifar10"), dataset_res=resolution,
                                num_gen=kwargs.get('fid_n', 5000), dataset_split=kwargs.get('fid_split', "train"),
                                z_dim=latent_size, num_workers=2,  # batch_size=256,
                                mode='legacy_tensorflow')
    fid_score = float(fid_score)

    n_mosaic_cells = 36
    mosaic_kwargs = {"nrow": 6, "normalize": True}
    rnd_gen = torch.Generator()
    rnd_gen.manual_seed(0)
    z_eval = torch.randn(n_mosaic_cells, latent_size, generator=rnd_gen).to(device)
    z_eval = z_eval.reshape(*z_eval.shape, 1, 1)

    path = os.path.join(kwargs['folder'], 'generated')
    if not os.path.exists(path):
        Path(path).mkdir()
    path = os.path.join(path, str(kwargs['model_id']))
    if not os.path.exists(path):
        Path(path).mkdir()
    gridname = kwargs.get('gridname', f"{kwargs['last_epoch']}_tmp.png")
    path_final = os.path.join(path, gridname)

    gen_imgs_eval = model.generator(z_eval)
    grid = make_grid(gen_imgs_eval.data, **mosaic_kwargs)
    save_image(grid, path_final)

    model.generator.train()
    model.discriminator.train()

    return {'fid': -fid_score}

def train(model, hyperparameters, optimizer, data_provider, first_epoch, last_epoch,
          max_epochs, device, **kwargs):
    print(f'{first_epoch=} ; {last_epoch=} ; {max_epochs=}')
    additional_kwargs = kwargs['additional_kwargs']
    kwargs.update(additional_kwargs)
    criterion = torch.nn.BCEWithLogitsLoss()

    data_loaders = data_provider.create_dataloaders()

    columns = ['epoch time', 'overall training time', 'epoch', 'g_loss', 'd_loss', '-FID']

    all_values = {'epoch': [], 'g_loss': [], 'd_loss': [], '-FID': []}

    print('Start training...')

    time_start = time.time()
    if_weight_avg = kwargs.get('gen_avg_params', None) is not None
    if if_weight_avg:
        additional_kwargs['gen_avg_params'] = kwargs['gen_avg_params']

    force_no_train = kwargs.get('force_no_train', False)
    for epoch in range(first_epoch, last_epoch):
        time_ep = time.time()

        if not force_no_train:
            train_res = _train_epoch_gan(device, data_loaders['train'], model, criterion, optimizer,
                                         hyperparameters=hyperparameters, **additional_kwargs)
        else:
            train_res = {'loss': -1, 'd_loss': -1}

        values = [epoch + 1 * int(not force_no_train), train_res['loss'], train_res['d_loss']]

        if epoch == last_epoch - 1:
            all_values['epoch'].append(epoch + 1 * int(not force_no_train))

            all_values['g_loss'].append(train_res['loss'])
            all_values['d_loss'].append(train_res['d_loss'])

            if if_weight_avg:
                normal_params = utils.general.swap_model_params(model.generator, kwargs['gen_avg_params'])

            val_res = _evaluate_gan(device, model, last_epoch=last_epoch, **kwargs)

            if if_weight_avg:
                all_values['gen_avg_params'] = utils.general.swap_model_params(model.generator, normal_params)

            all_values['-FID'].append(val_res['fid'])
            values += [val_res['fid']]
        else:
            values += [-1]

        overall_training_time = time.time() - time_start
        values = [time.time() - time_ep, overall_training_time] + values
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        print(table)

    del criterion
    del data_loaders
    gc.collect()

    return all_values