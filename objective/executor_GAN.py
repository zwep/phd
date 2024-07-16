# encoding: utf-8

import functools
import abc
import copy
import json
import numpy as np
import os

import torch

import torch.utils.data
import torch.nn.functional as F

import matplotlib.pyplot as plt

from helper import misc as hmisc
from helper_torch import misc as htmisc
from objective.executor_base import ExecutorBase
import helper.plot_fun as hplotf
import data_generator.Generic as data_gen_generic
import logging


class ExecutorGAN(ExecutorBase):
    def __init__(self, model_path=None, config_file=None, **kwargs):
        super().__init__(model_path=model_path, config_file=config_file, **kwargs)

        self.name_generator = 'state_generator'
        self.name_discriminator = 'state_discriminator'
        self.load_discriminator = kwargs.get('load_discriminator', True)

        """Load the generator"""
        self.generator_gradients = self.config_param['model']['config_gan'].get('generator_gradients')
        generator_choice = hmisc.get_nested(self.config_param, ['model', 'config_gan', 'generator_choice'])
        config_model = hmisc.get_nested(self.config_param, ['model', f'config_{generator_choice}'])
        self.generator = self.get_model(config_model=config_model, model_choice=generator_choice)
        self.set_requires_grad([self.generator], self.generator_gradients)

        # Reload weights
        reload_weights_generator_config = self.config_param['model']['config_gan'].get('reload_weights_generator_config')
        if reload_weights_generator_config.get('status', None):
            # Now an important assumptions comes...
            # That is that the weights of this path fit well with the current loaded model..
            # Well.. one way to find out
            path_to_weights = reload_weights_generator_config['model_path']
            dir_temp_weight = os.path.join(path_to_weights, self.name_temp_weights)
            dir_model_weight = os.path.join(path_to_weights, self.name_model_weights)
            self.generator = self.load_weights_model(model_obj=self.generator,
                                                     dir_temp_weight=dir_temp_weight,
                                                     dir_model_weight=dir_model_weight)
        else:
            generator_init_type = self.config_param['model']['config_gan'].get('generator_init_type', 'normal')
            htmisc.init_weights(self.generator, init_type=generator_init_type)

        """Load the discriminator"""
        if self.load_discriminator:
            discriminator_choice = hmisc.get_nested(self.config_param, ['model', 'config_gan', 'discriminator_choice'])
            config_model = hmisc.get_nested(self.config_param, ['model', f'config_{discriminator_choice}'])
            self.discriminator = self.get_model(config_model=config_model, model_choice=discriminator_choice)

            # Reload the discriminator
            reload_weights_discriminator_config = self.config_param['model']['config_gan'].get('reload_weights_discriminator_config')
            # Set the gradients...
            self.discriminator_gradients = self.config_param['model']['config_gan'].get('discriminator_gradients')
            self.set_requires_grad([self.discriminator], self.discriminator_gradients)

            # Reload weights
            if reload_weights_discriminator_config.get('status', False):
                path_to_weights = reload_weights_discriminator_config['model_path']
                dir_temp_weight = os.path.join(path_to_weights, self.name_temp_weights)
                dir_model_weight = os.path.join(path_to_weights, self.name_model_weights)
                self.discriminator = self.load_weights_model(model_obj=self.discriminator,
                                                             dir_temp_weight=dir_temp_weight,
                                                             dir_model_weight=dir_model_weight)
            else:
                discriminator_init_type = self.config_param['model']['config_gan'].get('discriminator_init_type', 'normal')
                htmisc.init_weights(self.discriminator, init_type=discriminator_init_type)

        # Check if we are using a Multiscale Gradient model..
        self.ind_MSG = self.config_param['model']['config_gan'].get('MSG', False)

        """Determine lambda factors for the losses"""
        # Used as scaling factor for the generator loss between target and prediced image
        lambda_generator_config = self.config_param['model']['config_gan'].get('lambda_generator', {})
        self.lambda_generator = self.get_lambda_weight_fun(lambda_generator_config)

        lambda_generator_additional_config = self.config_param['model']['config_gan'].get('lambda_generator_additional_config', {})
        self.lambda_generator_additional = self.get_lambda_weight_fun(lambda_generator_additional_config)

        # Used when we have xCNN layers in the model
        lambda_generator_xcnn = self.config_param['model']['config_gan'].get('lambda_generator_xcnn', {})
        self.lambda_generator_xcnn = self.get_lambda_weight_fun(lambda_generator_xcnn)
        lambda_discriminator_xcnn_weight = self.config_param['model']['config_gan'].get('lambda_discriminator_xcnn', {})
        self.lambda_discriminator_xcnn = self.get_lambda_weight_fun(lambda_discriminator_xcnn_weight)

        """Define all the losses"""
        # This is a dict that contains all the losses...
        if self.load_model_only is False:
            temp_loss_dict = self.get_loss_dict()
            # Just print the dict..
            self.loss_generator = temp_loss_dict.get('loss_generator', None)
            self.loss_generator_add = temp_loss_dict.get('loss_generator_add', None)
            self.loss_valid_discriminator = temp_loss_dict.get('loss_discriminator', None)

            """Define learning rates / optimizers"""
            lr_generator = self.config_param['model']['config_gan'].get('lr_generator', 0)
            if lr_generator is not None:
                self.config_param['optimizer']['config']['lr'] = lr_generator
            self.optimizer_generator = self.get_optimizer(model_parameters=self.generator.parameters(),
                                                          optimizer_config=self.config_param['optimizer'])

            lr_discriminator = self.config_param['model']['config_gan'].get('lr_discriminator', 0)
            if lr_discriminator is not None:
                self.config_param['optimizer']['config']['lr'] = lr_discriminator
            self.optimizer_disciminator = self.get_optimizer(model_parameters=self.discriminator.parameters(),
                                                             optimizer_config=self.config_param['optimizer'])

            self.scheduler_generator = self.get_lr_scheduler(self.optimizer_generator)
            self.scheduler_discriminator = self.get_lr_scheduler(self.optimizer_disciminator)

            # Create data generators
            num_workers = self.config_param.get('data').get('num_workers', 0)
            if self.debug:
                print('INFO - EXEC: \t number of workers ', num_workers)

            self.test_loader = self.get_data_loader('test', num_workers=num_workers)
            self.test_batch_size = self.test_loader.batch_size

            if self.inference is False:
                self.train_loader = self.get_data_loader('train', num_workers=num_workers)
                self.train_batch_size = self.train_loader.batch_size
                self.val_loader = self.get_data_loader('validation', num_workers=num_workers)
                self.val_batch_size = self.val_loader.batch_size

    def get_loss_dict(self):
        # Here we get a dictionary of loss objects.
        # Why a dict? Because it scales more easily with all the options that we have...
        loss_dict = {}

        # Get the generator loss
        loss_generator_name = self.config_param['model']['config_gan']['generator_loss']
        loss_generator_config = self.config_param['model']['config_gan'].get('generator_loss_config', {})
        loss_generator_config.setdefault("run", {})
        loss_generator_config.setdefault("obj", {})

        loss_generator = self.get_loss(loss_generator_name, options=loss_generator_config['obj'])
        loss_dict['loss_generator'] = loss_generator

        # Get the additional loss for the generator
        loss_generator_add_name = self.config_param['model']['config_gan'].get('generator_additional_loss', None)
        loss_generator_add_config = self.config_param['model']['config_gan'].get('generator_additional_loss_config', {})
        loss_generator_add_config.setdefault("run", {})
        loss_generator_add_config.setdefault("obj", {})

        loss_generator_add = self.get_loss(loss_generator_add_name, options=loss_generator_add_config['obj'])
        loss_dict['loss_generator_add'] = loss_generator_add

        # Get the discriminator losss
        discriminator_loss_name = self.config_param['model']['config_gan'].get('discriminator_loss', None)
        discriminator_loss_config = self.config_param['model']['config_gan'].get('discriminator_loss_config', {})
        discriminator_loss_config.setdefault("run", {})
        discriminator_loss_config.setdefault("obj", {})
        loss_valid_discriminator = self.get_loss(discriminator_loss_name, options=discriminator_loss_config['obj'])
        loss_dict['loss_discriminator'] = loss_valid_discriminator

        return loss_dict

    @abc.abstractmethod
    def get_data_generators(self, indicator_train):
        raise NotImplementedError

    def get_image_prediction(self, sel_batch=None, sel_item=None, **kwargs):
        n_batches = self.test_loader.__len__()
        n_item = self.test_loader.batch_size
        print('INFO - EXEC: \t test batch size: ', n_item)
        if sel_batch is None:
            sel_batch = np.random.randint(n_batches)
        if sel_item is None:
            sel_item = np.random.randint(n_item)

        # Not doing anything with batch or item number...
        batch_counter = 0
        self.generator.eval()  # IMPORTANT

        with torch.no_grad():  # IMPORTANT
            for container in self.test_loader:
                if self.masked:
                    X = container['input']
                    y = container['target']
                    mask = container['mask']
                    mask = mask.to(self.device)
                else:
                    X = container['input']
                    y = container['target']

                torch_input, torch_target = X.to(self.device), y.to(self.device)

                if self.ind_MSG:
                    torch_pred, multiscale_input = self.generator(torch_input)
                else:
                    multiscale_input = None
                    torch_pred = self.generator(torch_input)

                if self.masked:
                    torch_pred = torch_pred * mask

                if sel_batch == batch_counter:
                    break

                batch_counter += 1

        self.generator.train()  # IMPORTANT
        output = self.postproc_output(torch_input, torch_target, torch_pred, sel_item=sel_item)
        torch_input, torch_target, torch_pred, plot_augm = output

        return torch_input, torch_target, torch_pred, plot_augm

    def forward_discriminator(self, input_generator, output_generator, target_generator, multiscale_input=None,
                              epoch=999):
        conditional_input = self.config_param['model']['config_gan'].get('conditional', False)

        # multiscale_input = kwargs.get('multiscale_input', None)
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        if conditional_input:  # Or if input_generator is None
            input_discriminator_fake = torch.cat((input_generator, output_generator), 1)
        else:
            input_discriminator_fake = output_generator

        if multiscale_input is None:
            multiscale_input = []

        prediction_fake = self.discriminator(input_discriminator_fake.detach(), multiscale_input=copy.copy(multiscale_input))
        loss_discriminator_fake = self.loss_valid_discriminator(prediction_fake, target_is_truth=False)

        # print('Forward Discriminator fake input', prediction_fake)
        # Real
        multiscale_input_truth = []
        if conditional_input:  # Or if input_generator is None
            input_discriminator_truth = torch.cat((input_generator, target_generator), 1)
        else:
            input_discriminator_truth = target_generator

        if multiscale_input is None:
            multiscale_input_truth = []
        else:
            shape_multi_scale = [x.shape[-2:] for x in multiscale_input]
            multiscale_input_truth = [F.interpolate(target_generator, size=x, mode='bilinear', align_corners=False) for x in shape_multi_scale]

        prediction_real = self.discriminator(input_discriminator_truth, multiscale_input=copy.copy(multiscale_input_truth))
        loss_discriminator_real = self.loss_valid_discriminator(prediction_real, target_is_truth=True)

        # combine loss and calculate gradients
        # Dissect the loss for further analysis..
        # loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2

        # Regularization on the xCNN weights. Only happens when we have those weights of course
        list_children = list(self.discriminator.children())
        loss_xcnn = 0
        for layer in list_children:
            named_param = list(layer.named_parameters())
            len_param = len(named_param)
            if len_param:
                for param_name, x_param in named_param:
                    if 'xcnn_weights' in param_name:
                        corr = self.corrcoef(x_param.reshape(x_param.shape[0], -1))
                        temp_loss = torch.sum(torch.abs(corr - torch.eye(corr.shape[0]).to(self.device)))
                        if self.debug:
                            print(param_name)
                            print('Exec gan - loss xcnn', temp_loss)
                        loss_xcnn += temp_loss

        loss_xcnn = loss_xcnn * self.lambda_discriminator_xcnn(epoch)

        # PAN loss
        # # #
        # calc Parceptual Adversarial Loss
        # self.loss_PAN = 0
        # for (fake_i, real_i, lam) in zip(fake_inters, real_inters, self.pan_lambdas):
        #     self.loss_PAN += self.criterionPAN(fake_i, real_i) * lam
        #     # print(f'Backward D self.loss_PAN {self.loss_PAN.get_device()}')
        #
        # if self.loss_PAN.data > self.pan_mergin_m:
        #     loss_PAN = Variable(self.Tensor(np.array([0], dtype=np.float)), requires_grad=False)
        # else:
        #     loss_PAN = Variable(self.Tensor(np.array([self.pan_mergin_m], dtype=np.float)),
        #                         requires_grad=False) - self.loss_PAN
        #
        # loss_PAN = loss_PAN.to(self.gpu_ids[0])
        # # print(f'Backward D loss_PAN {loss_PAN.get_device()}')
        #
        # # Combined loss
        # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + loss_PAN
        #
        # # #
        return loss_discriminator_real, loss_discriminator_fake, loss_xcnn

    def forward_generator(self, target_generator, output_generator, input_generator=None,
                          multiscale_input=None, epoch=999):

        conditional_input = self.config_param['model']['config_gan'].get('conditional', False)

        if conditional_input:  # Or if input_generator is None
            input_discriminator_fake = torch.cat((input_generator, output_generator), 1)
        else:
            input_discriminator_fake = output_generator

        if multiscale_input is None:
            multiscale_input = []

        prediction_fake = self.discriminator(input_discriminator_fake, multiscale_input=copy.copy(multiscale_input))
        # print('Forward Generator fake input', prediction_fake)
        # I kinda understand the True value here.. but it feels weird man.
        loss_discriminator_fake = self.loss_valid_discriminator(prediction_fake, target_is_truth=True)
        # Second, G(A) = B
        loss_generator_L1 = self.loss_generator(output_generator, target_generator) * self.lambda_generator(epoch)
        if self.loss_generator_add:
            loss_generator_add = self.loss_generator_add(output_generator, target_generator) * self.lambda_generator_additional(epoch)
        else:
            loss_generator_add = 0

        # Removed this to dissect the loss and store its components..
        # loss_generator = loss_discriminator_fake + loss_generator_L1

        # Regularization on the xCNN weights. Only happens when we have those weights of course
        list_children = list(self.generator.children())
        loss_xcnn = 0
        for layer in list_children:
            named_param = list(layer.named_parameters())
            len_param = len(named_param)
            if len_param:
                for param_name, x_param in named_param:
                    if 'xcnn_weights' in param_name:
                        corr = self.corrcoef(x_param.reshape(x_param.shape[0], -1))
                        temp_loss = torch.sum(torch.abs(corr - torch.eye(corr.shape[0]).to(self.device)))
                        if self.debug:
                            print(param_name)
                            print('Exec gan - loss xcnn', temp_loss)
                        loss_xcnn += temp_loss

        loss_xcnn = loss_xcnn * self.lambda_generator_xcnn(epoch)

        return loss_discriminator_fake, loss_generator_L1, loss_generator_add, loss_xcnn

    def save_history_graph(self):
        doutput = self.config_param['dir']['doutput']
        dir_history_graph = os.path.join(doutput, self.name_loss_graph)
        dir_history_graph_comp = os.path.join(doutput, self.name_loss_component_graph)

        n = max(len(self.history_dict['train_loss']) - 1, 1)

        # Overview of validation, train, and test loss in one graph.
        fig_handle, ax = plt.subplots()
        ax.plot(self.history_dict['val_loss'], 'b-x', label='validation loss', alpha=0.5)
        ax.plot(self.history_dict['train_loss'], 'r-x', label='training loss', alpha=0.5)
        # n_test_cases = len(self.history_dict['train_loss'])
        ax.scatter(n, self.history_dict['test_loss'], c='k', label='test loss', alpha=0.5)
        ax.set_title('validation loss vs train loss vs test loss')
        plt.legend()

        # Plotting of individual components of the loss....
        fig_handle_deux, ax = plt.subplots()
        plot_array_generator = hmisc.change_list_order(self.history_dict['train_generator_loss'])
        generator_names = ['gen fake loss', 'generator loss', 'gen add. loss', 'xcnn']
        for i, i_plot in enumerate(plot_array_generator):
            ax.plot(i_plot, label=generator_names[i], alpha=0.8 ** i)

        plot_array_discriminator = hmisc.change_list_order(self.history_dict['train_discriminator_loss'])
        discriminator_names = ['real loss', 'fake loss', 'xcnn']
        for i, i_plot in enumerate(plot_array_discriminator):
            ax.plot(i_plot, label=discriminator_names[i], alpha=0.8 ** i)

        plt.legend()
        fig_handle.savefig(dir_history_graph)
        fig_handle_deux.savefig(dir_history_graph_comp)

        return fig_handle

    def train_model(self):
        # https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/acgan/acgan.py
        # https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py

        self.discriminator.train()
        self.generator.train()

        generator_loss_curve = []
        discriminator_loss_curve = []
        train_loss_curve = []
        val_loss_curve = []

        return_gradients = self.config_param['model'].get('return_gradients', False)

        if return_gradients:
            self.history_dict.setdefault('overal_grad_gen', [])
            self.history_dict.setdefault('overal_param_gen', [])
            self.history_dict.setdefault('overal_grad_discr', [])
            self.history_dict.setdefault('overal_param_discr', [])

        n_epoch = self.config_param['model']['n_epoch']
        n_discriminator_training = self.config_param['model']['config_gan'].get('n_discriminator_training', 1)
        # Clip weight setting - generator
        ind_weight_clip_discr = self.config_param['model']['config_gan'].get('discriminator_clipweights', False)
        weight_clip_discr = self.config_param['model']['config_gan'].get('discriminator_clipweights_value', 0)

        # Clip weight setting - generator
        ind_weight_clip_gen = self.config_param['model']['config_gan'].get('generator_clipweights', False)
        weight_clip_gen = self.config_param['model']['config_gan'].get('generator_clipweights_value', 0)

        epoch = 0
        breakdown_counter = 0
        min_epoch_loss = 9999

        if self.debug:
            print('')
        while epoch < n_epoch and breakdown_counter < self.breakdown_limit:
            try:
                print(f"Epoch {epoch + 1}/{n_epoch} ...", end='')
                print(f'Breakdown counter {breakdown_counter}')

                epoch_loss = 0

                epoch_generator_loss = np.array([0, 0, 0, 0])
                epoch_discriminator_loss = np.array([0, 0, 0])

                for i, container in enumerate(self.train_loader):
                    if self.masked:
                        X = container['input']
                        y = container['target']
                        mask = container['mask']
                        mask = mask.to(self.device)
                    else:
                        X = container['input']
                        y = container['target']

                    X = X.to(self.device)
                    y = y.to(self.device)

                    if self.debug:
                        print('Input tensor on device size ', X.shape, X.device)
                        print('Target tensor on device size ', y.shape, X.device)

                    if self.ind_MSG:
                        y_pred, multiscale_input = self.generator(X)
                    else:
                        multiscale_input = None
                        if self.generator_gradients is False:
                            with torch.no_grad():
                                y_pred = self.generator(X)
                        else:
                            y_pred = self.generator(X)

                    if self.masked:
                        y_pred = y_pred * mask
                        # Just to be sure that we are the same....
                        y = y * mask

                    # update D
                    if self.discriminator_gradients is False:
                        self.set_requires_grad(network=self.discriminator, requires_grad=False)
                    else:
                        self.set_requires_grad(network=self.discriminator,  requires_grad=True)
                    # # #
                    for _ in range(n_discriminator_training):
                        self.optimizer_disciminator.zero_grad()  # set D's gradients to zero
                        loss_components_discriminator = self.forward_discriminator(target_generator=y,
                                                                                   output_generator=y_pred,
                                                                                   input_generator=X,
                                                                                   multiscale_input=copy.copy(multiscale_input),
                                                                                   epoch=epoch)
                        loss_real, loss_fake, loss_xcnn_discr = loss_components_discriminator
                        loss_discriminator = (loss_real + loss_fake)/2
                        if self.config_param["model"]["config_gan"]["discriminator_gradients"]:
                            loss_discriminator.backward()
                            self.optimizer_disciminator.step()  # udpate G's weights

                        # See https://paper.dropbox.com/doc/Wasserstein-GAN-GvU0p2V9ThzdwY3BbhoP7
                        # For more information. Quite nice.
                        # Maybe check this one for f-divergence stuff https://arxiv.org/pdf/1606.00709.pdf
                        if ind_weight_clip_discr:
                            for p in self.discriminator.parameters():
                                p.data.clamp_(-weight_clip_discr, weight_clip_discr)

                    # update G
                    # D requires no gradients when optimizing G
                    self.set_requires_grad(network=self.discriminator, requires_grad=False)
                    self.optimizer_generator.zero_grad()  # set G's gradients to zero
                    loss_components_generator = self.forward_generator(target_generator=y,
                                                                       output_generator=y_pred,
                                                                       input_generator=X,
                                                                       multiscale_input=multiscale_input,
                                                                       epoch=epoch)  # calculate graidents for G

                    loss_discriminator_fake, loss_generator_L1, loss_add, loss_xcnn_gen = loss_components_generator
                    loss_generator = loss_discriminator_fake + loss_generator_L1 + loss_xcnn_gen + loss_add
                    if self.config_param["model"]["config_gan"]["generator_gradients"]:
                        loss_generator.backward()
                        self.optimizer_generator.step()  # udpate G's weights

                    if ind_weight_clip_gen:
                        for p in self.generator.parameters():
                            p.data.clamp_(-weight_clip_gen, weight_clip_gen)

                    # # # Store the different losses

                    if hasattr(loss_xcnn_gen, 'item'):
                        loss_xcnn_gen = loss_xcnn_gen.item()

                    if hasattr(loss_add, 'item'):
                        loss_add = loss_add.item()

                    temp_generator_loss = np.array([loss_discriminator_fake.item(),
                                                    loss_generator_L1.item(),
                                                    loss_add,
                                                    loss_xcnn_gen])
                    epoch_generator_loss = epoch_generator_loss + temp_generator_loss

                    #if self.debug:
                    # We are just going to present this.. always
                    print('\tGenerator loss components ', [x.item() for x in temp_generator_loss], end='\r')

                    if self.debug:
                        if self.loss_generator_add:
                            if hasattr(self.loss_generator_add, 'n'):
                                print('Additional loss state ', self.loss_generator_add.n, self.loss_generator_add.internal_counter)
                            if hasattr(self.loss_generator_add, 'n_mask'):
                                print('Additional loss state ', self.loss_generator_add.n_mask, self.loss_generator_add.internal_counter)

                    if hasattr(loss_xcnn_discr, 'item'):
                        loss_xcnn_discr = loss_xcnn_discr.item()

                    temp_discriminator_loss = np.array([loss_real.item(),
                                                        loss_fake.item(),
                                                        loss_xcnn_discr])

                    epoch_discriminator_loss = epoch_discriminator_loss + temp_discriminator_loss
                    epoch_loss += loss_discriminator.item()

                    # Store the gradients...
                    if return_gradients:
                        list_children_generator = htmisc.get_all_children(self.generator, [])
                        list_children_discriminator = htmisc.get_all_children(self.discriminator, [])
                        param_per_layer_gen, grad_per_layer_gen = self.get_param_gradients(list_children_generator)
                        param_per_layer_discr, grad_per_layer_discr = self.get_param_gradients(list_children_discriminator)

                        self.history_dict['overal_grad_gen'].append(grad_per_layer_gen)
                        self.history_dict['overal_param_gen'].append(param_per_layer_gen)

                        self.history_dict['overal_grad_discr'].append(grad_per_layer_discr)
                        self.history_dict['overal_param_discr'].append(param_per_layer_discr)


                # Update the scheduler
                self.scheduler_generator.step()
                self.scheduler_discriminator.step()

                lr_gen = self.optimizer_generator.param_groups[0]['lr']
                lr_discr = self.optimizer_disciminator.param_groups[0]['lr']
                print('learning rate generator = %.7f' % lr_gen)
                print('learning rate discriminator = %.7f' % lr_discr)

                # Validate the model
                val_loss_value = self.validate_model()
                val_loss_curve.append(val_loss_value)

                # Average over amount of batches..
                n_training = self.train_loader.__len__()
                epoch_loss = epoch_loss / n_training
                epoch_discriminator_loss = epoch_discriminator_loss / n_training
                epoch_generator_loss = epoch_generator_loss / n_training

                discriminator_loss_curve.append(epoch_discriminator_loss)
                generator_loss_curve.append(epoch_generator_loss)
                train_loss_curve.append(epoch_loss)

                print('average training loss over epoch: ', epoch_loss)

                if epoch_loss < min_epoch_loss:
                    if self.debug:
                        print('INFO - EXEC: \t Old minimum loss ', min_epoch_loss)
                        print('INFO - EXEC: \t New minimum loss ', epoch_loss)
                    min_epoch_loss = epoch_loss
                    self.temp_weights = {self.name_generator: self.generator.state_dict(),
                                         self.name_discriminator: self.discriminator.state_dict()}

                # We want to have a proper loss...
                if epoch_loss != 0:
                    # For all the non-zero losses...
                    temp_curve = [x for x in val_loss_curve if x != 0 and x is not None]
                    historic_curve = temp_curve[-(self.memory_time + self.memory_length): -self.memory_time]
                    if len(historic_curve):
                        historic_loss = np.mean(historic_curve)
                    else:
                        # Some big number....
                        historic_loss = 9999

                    current_loss = np.mean(temp_curve[-self.memory_length:])
                    criterion = historic_loss - current_loss
                    print(f'validation loss criterion {criterion}\n'
                          f'historic loss {historic_loss} \n'
                          f'validation loss {val_loss_value}')
                    # If the current loss gets larger than the historic loss...
                    if criterion < 0:
                        breakdown_counter += 1
                        if self.debug:
                            print('INFO - EXEC: \t Not improved, counter: ', breakdown_counter, self.breakdown_limit)
                    else:
                        breakdown_counter -= 1
                        breakdown_counter = max(breakdown_counter, 0)
                        if self.debug:
                            print('INFO - EXEC: \t Improved! Compared to', historic_loss, current_loss)

                epoch += 1

                # Every now and then.. save intermediate results..
                if epoch % np.ceil(0.10 * n_epoch) == 0:
                    doutput = self.config_param['dir']['doutput']
                    self.temp_weights = {self.name_generator: self.generator.state_dict(),
                                         self.name_discriminator: self.discriminator.state_dict()}
                    self.save_weight_state_dict(self.temp_weights, path=doutput, name=self.name_temp_weights)

                    self.save_model(plot_name='intermediate')
                    plt.close('all')

            except KeyboardInterrupt:
                print('\t\t Keyboard interrupt ')
                break

        # Messages
        if breakdown_counter > self.breakdown_limit:
            print('INFO - EXEC: \t We are not advancing fast enough. Broke out of training loop')
        elif epoch >= n_epoch:
            print('INFO - EXEC: \t Completed all epochs')
        else:
            print('INFO - EXEC: \t Increase in loss.. break down')

        self.history_dict['train_loss'] = train_loss_curve  # batch_loss_curve
        self.history_dict['train_generator_loss'] = [list(x) for x in generator_loss_curve]  # batch_loss_curve
        self.history_dict['train_discriminator_loss'] = [list(x) for x in discriminator_loss_curve]
        self.history_dict['val_loss'] = val_loss_curve

        return self.history_dict

    def test_model(self):
        # Not sure yet why this is different from val_model.. but w/e
        self.generator.eval()
        self.discriminator.eval()

        loss = []
        with torch.no_grad():  # IMPORTANT
            for container in self.test_loader:
                if self.masked:
                    X = container['input']
                    y = container['target']
                    mask = container['mask']
                    mask = mask.to(self.device)
                else:
                    X = container['input']
                    y = container['target']

                X = X.to(self.device)
                y = y.to(self.device)

                if self.ind_MSG:
                    y_pred, multiscale_input = self.generator(X)
                else:
                    multiscale_input = None
                    y_pred = self.generator(X)

                if self.masked:
                    y_pred = y_pred * mask
                    # Just to be sure that we are the same....
                    y = y * mask

                # Here we need to process the predictions according to the losses...
                # For now we have some old code laying here..
                loss_real, loss_fake, loss_xcnn_discr = self.forward_discriminator(input_generator=X,
                                                                                   target_generator=y,
                                                                                   output_generator=y_pred,
                                                                                   multiscale_input=multiscale_input)
                loss_discriminator = (loss_real + loss_fake) / 2
                loss_item = loss_discriminator.item()  # This is of course not trivial in a GAN...
                loss.append(loss_item)

        self.generator.train()
        self.discriminator.train()

        self.test_loss = sum(loss)/len(loss)
        self.history_dict['test_loss'] = self.test_loss
        return loss

    def validate_model(self):
        self.generator.eval()
        self.discriminator.eval()

        loss = 0
        with torch.no_grad():  # IMPORTANT
            for container in self.val_loader:
                if self.masked:
                    X = container['input']
                    y = container['target']
                    mask = container['mask']
                    mask = mask.to(self.device)
                else:
                    X = container['input']
                    y = container['target']

                X = X.to(self.device)
                y = y.to(self.device)

                if self.ind_MSG:
                    y_pred, multiscale_input = self.generator(X)
                else:
                    multiscale_input = None
                    y_pred = self.generator(X)

                if self.masked:
                    y_pred = y_pred * mask
                    # Just to be sure that we are the same....
                    y = y * mask

                # Here we need to process the predictions according to the losses...
                # For now we have some old code laying here..
                loss_real, loss_fake, loss_xcnn_discr = self.forward_discriminator(input_generator=X,
                                                                target_generator=y,
                                                                output_generator=y_pred,
                                                                multiscale_input=multiscale_input)
                loss_discriminator = (loss_real + loss_fake) / 2
                loss += loss_discriminator.item()  # This is of course not trivial in a GAN...

        loss = loss / self.val_loader.__len__()
        self.generator.train()
        self.discriminator.train()

        return loss

    @staticmethod
    def save_weight_state_dict(x, path, name):
        for prefix_name, state_dict in x.items():
            print('Storing weights for ', prefix_name)
            torch.save(state_dict, os.path.join(path, prefix_name + '_' + name + '.pt'))
            # torch.save(self.model_obj.state_dict(), dir_weights)

    @staticmethod
    def corrcoef(x):
        # Used only for the xCNN weight compensation in the forward_generator function
        mean_x = torch.mean(x, dim=1, keepdim=True)
        xm = x.sub(mean_x.expand_as(x))
        c = xm.mm(xm.t())
        c = c / (x.size(1) - 1)
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c / stddev[:, None]
        c = c / stddev[None, :]
        c = torch.clamp(c, -1.0, 1.0)
        return c

    def save_model(self, plot_name='final'):
        # Here we are storing the model weights...
        # Where are we saving the temp weights?
        doutput = self.config_param['dir']['doutput']
        dir_history = os.path.join(self.config_param['dir']['doutput'], self.name_model_hist)

        model_weights = {self.name_generator: self.generator.state_dict(),
                         self.name_discriminator: self.discriminator.state_dict()}

        self.save_weight_state_dict(model_weights, path=doutput, name=self.name_model_weights)
        self.save_weight_state_dict(self.temp_weights, path=doutput, name=self.name_temp_weights)

        # Save history object
        serialized_json_history_obj = json.dumps(self.history_dict)
        with open(dir_history, 'w') as f:
            f.write(serialized_json_history_obj)

        # New code for plotting results...
        x0, y0, y_pred0, plot_augmentation = self.get_image_prediction()
        print('Executor GAN shape of img pred ', x0.shape, y0.shape, y_pred0.shape)
        if x0.ndim > 2:
            # For now.. just assume that if we have a larger dimension.. then the amount of channels will always be in the first dim.
            n_chan = x0.shape[0]
        else:
            n_chan = -1
            print('Unkown output dimension of x0 ', x0.shape)

        # Pick one batch...
        try:
            temp_division = y_pred0[0:1]/(x0+1e-4)
            temp_division[temp_division < -1] = -1
            temp_division[temp_division > 1] = 1
        except ValueError:
            print('Division was not possible.')
            temp_division = np.zeros((10, 10))


        # This is a really dirty trick to fix this issue...
        # Issue: stacking with shapes like (8, 256, 256) and (1, 256, 256) is not possible
        # But my function also accepts lists
        try:
            plot_array = np.stack([x0, y0, y_pred0, temp_division], axis=0)
            print('Shape of plot array', plot_array.shape)
        except:
            plot_array = [x0, y0, y_pred0, temp_division]
            for iplot in plot_array:
                print('Shape of plot array', iplot.shape)

        # subtitle = [['input'] * n_chan, ['target'] * n_chan, ['pred'] * n_chan, ['division'] * n_chan]
        subtitle = None
        for i_augm in plot_augmentation:
            output_path = os.path.join(doutput, f"{plot_name}_{i_augm}.jpg")
            fig_handle = hplotf.plot_3d_list(plot_array, figsize=(15, 10), dpi=75, augm=i_augm,
                                             subtitle=subtitle,
                                             title=i_augm)
            fig_handle.savefig(output_path)

        del plot_array  # Frees up a lot of space...

    def load_weights(self, dir_temp_weight=None, dir_model_weight=None):
        #
        if self.load_discriminator:
            temp_names = [self.name_generator]  # Actually only need two...
        else:
            temp_names = [self.name_generator, self.name_discriminator]  # Actually only need two...

        doutput = self.config_param['dir']['doutput']
        if dir_temp_weight is None:
            dir_temp_weight_dict = {x: os.path.join(doutput, x + '_' + self.name_temp_weights) for x in temp_names}
        else:
            dir_temp_weight_dict = {}

        if dir_model_weight is None:
            dir_model_weight_dict = {x: os.path.join(doutput, x + '_' + self.name_model_weights) for x in temp_names}
        else:
            dir_model_weight_dict = {}

        bool_all_temp_weights = all([os.path.isfile(v) for k, v in dir_temp_weight_dict.items()])
        bool_all_model_weights = all([os.path.isfile(v) for k, v in dir_model_weight_dict.items()])

        if bool_all_temp_weights:
            print('Loading temp weights')
            weight_generator = torch.load(dir_temp_weight_dict[self.name_generator], map_location=self.device)
            self.generator.load_state_dict(weight_generator)
            if self.load_discriminator is False:
                weight_discriminator = torch.load(dir_temp_weight_dict[self.name_discriminator], map_location=self.device)
                self.discriminator.load_state_dict(weight_discriminator)
        elif bool_all_model_weights:
            print('Loading model weights')
            weight_generator = torch.load(dir_model_weight_dict[self.name_generator], map_location=self.device)
            self.generator.load_state_dict(weight_generator)
            if self.load_discriminator is False:
                weight_discriminator = torch.load(dir_model_weight_dict[self.name_discriminator], map_location=self.device)
                self.discriminator.load_state_dict(weight_discriminator)
        else:
            print('ERROR - EXEC: \t Please, there is no weight file yet')
            print('\t searched ', doutput)

