import traceback

from copy import deepcopy

import numpy as np
import torch
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm

import search.algorithms.mix
import utils
import utils.general
from search.gan.train_gan import train
from utils.general import adjust_optimizer_settings, optimizer_to


class GanFitness:
    def __init__(self, folder, max_epochs, epoch_step, data_provider, hyperparameters, function_create_model,
                 args, shrink_perturb_always=None, if_gen_avg=False):
        self.folder = folder
        self.max_epochs = max_epochs
        self.epoch_step = epoch_step
        self.hyperparameters = hyperparameters
        self.data_provider = data_provider
        self.function_create_model = function_create_model
        self.device = 'cuda'
        self.train_kwargs = {}

        self.train_kwargs['latent_dim'] = args.model_parameters['latent_dim']
        if 'generator_batch_size_multiplier' in args.model_parameters:
            self.train_kwargs['generator_batch_size_multiplier'] = args.model_parameters['generator_batch_size_multiplier']

        # annoyingly, cleanfid wants custom dataset to have "custom" as split,
        # which means that actual split has to be encoded in the name
        self.fid_dataset_train = {'cifar10advnas': 'cifar10', 'stl10advnas': 'stl10-train'}[args.dataset_name]
        self.fid_split_train = {'cifar10advnas': 'train', 'stl10advnas': 'custom'}[args.dataset_name]
        self.dataset_name = args.dataset_name

        self.shrink_perturb_always = shrink_perturb_always
        self.if_gen_avg = if_gen_avg

    def fitness(self, encoded_solution, model_id, epoch, save_at_epoch_0=False, **kwargs):
        try:
            encoded_solution, net, optimizer, avg_gen_params = self.create_model_and_optimizer(encoded_solution,
                                                                                 f'models/model_{model_id}_{epoch}')

            if save_at_epoch_0 and epoch == 0:
                dict_to_save = {
                    'model_state_dict': net.state_dict(),
                    'hyperparameters': self.hyperparameters,
                    'optimizer_state_dict': None,
                }
                if self.if_gen_avg:
                    normal_params = utils.general.swap_model_params(net.generator, avg_gen_params)
                    dict_to_save['avg_gen_state_dict'] = deepcopy(net.generator.state_dict())
                    avg_gen_params = utils.general.swap_model_params(net.generator, normal_params)

                torch.save(dict_to_save, '%s/models/model_%d_%d' % (self.folder, model_id, epoch))

            if self.shrink_perturb_always is not None:
                if epoch != 0:
                    model_for_perturb_state_dict = self.function_create_model(data_provider=self.data_provider,
                                                                              hyperparameters=self.hyperparameters).to(self.device).state_dict()
                    net_state = search.algorithms.mix.shrink_perturb_whole_state(net.state_dict(), self.shrink_perturb_always, model_for_perturb_state_dict)
                    net.load_state_dict(net_state)
                else:
                    print('skipping shrink-perturb for epoch 0')

            info_train_and_val = train(net, self.hyperparameters, optimizer, self.data_provider, epoch,
                                       epoch + self.epoch_step, self.max_epochs, self.device,
                                       model_id=model_id, folder=self.folder,
                                       additional_kwargs=self.train_kwargs, fid_dataset=self.fid_dataset_train,
                                       fid_split=self.fid_split_train, gen_avg_params=avg_gen_params)
            val_score = info_train_and_val['-FID']

            dict_to_save = {'cur_scores': [val_score], 'model_state_dict': net.state_dict(),
                            'hyperparameters': self.hyperparameters,
                            'optimizer_generator_state_dict': optimizer['generator'].state_dict(),
                            'optimizer_discriminator_state_dict': optimizer['discriminator'].state_dict()}

            if self.if_gen_avg:
                avg_gen_params = info_train_and_val['gen_avg_params']
                normal_params = utils.general.swap_model_params(net.generator, avg_gen_params)
                dict_to_save['avg_gen_state_dict'] = deepcopy(net.generator.state_dict())
                _ = utils.general.swap_model_params(net.generator, normal_params)

            torch.save(dict_to_save, '%s/models/model_%d_%d' % (self.folder, model_id, epoch + self.epoch_step))

            torch.cuda.empty_cache()
            return val_score[-1]

        except Exception as e:
            print(f'Exception occured: {encoded_solution=} {model_id=} {epoch=}')
            print(traceback.format_exc())
            raise e

    def create_model_and_optimizer(self, encoded_solution, checkpoint_name):
        print('encoded_solution', utils.general.seq_to_str(encoded_solution))
        encoded_solution = [int(x) for x in encoded_solution]
        encoded_solution = tuple(encoded_solution)
        self.hyperparameters.convert_encoding_to_hyperparameters(encoded_solution)

        net = self.function_create_model(data_provider=self.data_provider, hyperparameters=self.hyperparameters)

        checkpoint = utils.general.try_load_checkpoint(self.folder, checkpoint_name)

        if not self.if_gen_avg:
            if checkpoint is not None:
                net.load_state_dict(checkpoint['model_state_dict'])
            avg_gen_params = None
        else:
            avg_gen_net = deepcopy(net.generator)
            if checkpoint is not None:
                net.load_state_dict(checkpoint['model_state_dict'])
                avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
            avg_gen_net.to(self.device)
            avg_gen_params = utils.general.copy_params(avg_gen_net)
            del avg_gen_net

        opt_g = self.hyperparameters.get_optimizer(net.generator, lr_key='lr_g')
        opt_d = self.hyperparameters.get_optimizer(net.discriminator, lr_key='lr_d')
        optimizer = {'generator': opt_g, 'discriminator': opt_d}

        if checkpoint is not None:
            if 'optimizer_generator_state_dict' in checkpoint:
                try:
                    optimizer['generator'].load_state_dict(checkpoint['optimizer_generator_state_dict'])
                except:
                    print('Failed to load generator optimizer state')
            if 'optimizer_discriminator_state_dict' in checkpoint:
                try:
                    optimizer['discriminator'].load_state_dict(checkpoint['optimizer_discriminator_state_dict'])
                except:
                    print('Failed to load discriminator optimizer state')

            lr, wd = self.hyperparameters.get_optimizer_params(lr_key='lr_g')
            print(f'Optimizer (loaded), generator: {lr=} ; {wd=}')
            optimizer['generator'] = adjust_optimizer_settings(optimizer['generator'], lr=lr, wd=wd)

            lr, wd = self.hyperparameters.get_optimizer_params(lr_key='lr_d')
            print(f'Optimizer (loaded), discriminator: {lr=} ; {wd=}')
            optimizer['discriminator'] = adjust_optimizer_settings(optimizer['discriminator'], lr=lr, wd=wd)
            optimizer_to(optimizer, self.device)

        net = net.to(self.device)
        return encoded_solution, net, optimizer, avg_gen_params


    def evaluate_final_gan_FID(self, encoded_solution, checkpoint_name, fid_n, fid_split):
        encoded_solution, net, optimizer, avg_gen_params = self.create_model_and_optimizer(encoded_solution, checkpoint_name)
        info_train_and_val = train(net, self.hyperparameters, optimizer, self.data_provider, 171717,
                                   171717 + 1, None, self.device,
                                   model_id=checkpoint_name, folder=self.folder,
                                   additional_kwargs=self.train_kwargs, force_no_train=True,
                                   fid_n=fid_n, fid_split=self.fid_split_train, fid_dataset=self.fid_dataset_train,
                                   gen_avg_params=avg_gen_params)
        return info_train_and_val['-FID'][0]

    def evaluate_final_gan_inception(self, encoded_solution, checkpoint_name):
        encoded_solution, model, optimizer, avg_gen_params = self.create_model_and_optimizer(encoded_solution, checkpoint_name)

        if self.if_gen_avg:
            normal_params = utils.general.swap_model_params(model.generator, avg_gen_params)

        eval_batch_size = 128
        eval_iter = 50000 // eval_batch_size
        torch.cuda.empty_cache()
        model.generator.eval()

        inception = InceptionScore().cuda()

        for _ in tqdm(range(eval_iter), desc='sample images'):
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (eval_batch_size, self.train_kwargs['latent_dim'])))
            gen_imgs = model.generator(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).detach().type(torch.uint8)
            inception.update(gen_imgs)

        del model, optimizer, avg_gen_params
        torch.cuda.empty_cache()

        mean, std = inception.compute()
        mean, std = mean.item(), std.item()

        return mean, std