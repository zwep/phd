# encoding: utf-8
import abc
import itertools
import json
import functools
import numpy as np
import os
import torch

import helper.plot_fun as hplotf

from helper import misc as hmisc
from helper_torch import misc as htmisc
from model.VGG import Vgg16
from objective.executor_base import ExecutorBase


import model.CycleGAN as model_cyclegan
import model.DeepDenseNet as model_deepdense
import model.UNet as model_unet
import model.UNetSkipConnection as model_unetskip
import model.XNet as model_xnet
import model.XXNet as model_xxnet
import model.XXXNet as model_xxxnet
import model.GroupedXNet as model_xnet_group
import model.BridgeXNet as model_xnet_bridge
import model.ResNet as model_resnet
import model.YNet as model_ynet
import matplotlib.pyplot as plt


class ExecutorCycleGAN(ExecutorBase):
    def __init__(self, model_path=None, config_file=None, **kwargs):
        super().__init__(model_path=model_path, config_file=config_file)

        # #
        # This is super ugly... I could cut it into pieces....
        # #
        self.name_A2B = 'state_A2B'
        self.name_B2A = 'state_B2A'
        self.name_D_A = 'state_DA'
        self.name_D_B = 'state_DB'

        self.netG_A2B = self.get_generator('A')
        self.netG_B2A = self.get_generator('B')

        self.netD_A = self.get_generator('A')
        self.netD_B = self.get_generator('B')

        """
        Get the losses..
        """
        # Waarom deed je nou niet gewoon ff die naam erbij in de config..?
        GAN_loss_name = self.config_param['model']['config_cyclegan'].get('GAN_loss', None)
        GAN_loss_config = self.config_param['model']['config_cyclegan'].get('GAN_loss_config', {})
        GAN_loss_config.setdefault("run", {})
        GAN_loss_config.setdefault("obj", {})
        self.criterion_GAN = self.get_loss(GAN_loss_name, options=GAN_loss_config['obj'])

        cycle_loss_name = self.config_param['model']['config_cyclegan'].get('cycle_loss', None)
        cycle_loss_config = self.config_param['model']['config_cyclegan'].get('cycle_loss_config', {})
        cycle_loss_config.setdefault("run", {})
        cycle_loss_config.setdefault("obj", {})
        self.criterion_cycle = self.get_loss(cycle_loss_name, options=cycle_loss_config['obj'])

        lambda_cycle_config = self.config_param['model']['config_cyclegan'].get('lambda_cycle', {})
        self.lambda_cycle = self.get_lambda_weight_fun(lambda_cycle_config)

        identity_loss_name = self.config_param['model']['config_cyclegan'].get('identity_loss', None)
        identity_loss_config = self.config_param['model']['config_cyclegan'].get('identity_loss_config', {})
        identity_loss_config.setdefault("run", {})
        identity_loss_config.setdefault("obj", {})
        self.criterion_identity = self.get_loss(identity_loss_name, options=identity_loss_config['obj'])

        lambda_identity_config = self.config_param['model']['config_cyclegan'].get('lambda_identity', {})
        self.lambda_identity = self.get_lambda_weight_fun(lambda_identity_config)

        """
        Get the optimizers..
        """
        netG_parameters = itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters())
        self.optimizer_G = self.get_optimizer(netG_parameters,
                                              optimizer_config=self.config_param['optimizer'])
        self.optimizer_D_A = self.get_optimizer(self.netD_A.parameters(),
                                                optimizer_config=self.config_param['optimizer'])
        self.optimizer_D_B = self.get_optimizer(self.netD_B.parameters(),
                                                optimizer_config=self.config_param['optimizer'])

        self.lr_scheduler_G = self.get_lr_scheduler(self.optimizer_G)
        self.lr_scheduler_D_A = self.get_lr_scheduler(self.optimizer_D_A)
        self.lr_scheduler_D_B = self.get_lr_scheduler(self.optimizer_D_B)

        # Create data generators
        num_workers = self.config_param.get('data').get('num_workers', 16)
        if self.debug:
            print('INFO - EXEC: \t number of workers ', num_workers)

        # Create train/test loader
        self.test_loader = self.get_data_loader('test', num_workers=num_workers)
        self.test_batch_size = self.test_loader.batch_size

        # No need to use these when we only want to use the test set..
        if self.inference is False:
            self.train_loader = self.get_data_loader('train', num_workers=num_workers)
            self.train_batch_size = self.train_loader.batch_size

            self.val_loader = self.get_data_loader('validation', num_workers=num_workers)
            self.val_batch_size = self.val_loader.batch_size

    def get_generator(self, type):
        # Get generator `type`
        generator_choice = hmisc.get_nested(self.config_param, ['model', 'config_gan', f'generator_{type}_choice'])
        config_model = hmisc.get_nested(self.config_param, ['model', f'config_{generator_choice}'])
        netG = self.get_model(config_model=config_model, model_choice=generator_choice)
        reload_weights_generator_config = self.config_param['model']['config_cyclegan'].get(f'reload_weights_generator_{type}_config', {})
        generator_gradients = self.config_param['model']['config_cyclegan'].get(f'generator_{type}_gradients', True)
        self.set_requires_grad([netG], generator_gradients)
        if reload_weights_generator_config.get('status', None):
            path_to_weights = reload_weights_generator_config['model_path']
            dir_temp_weight = os.path.join(path_to_weights, self.name_temp_weights)
            dir_model_weight = os.path.join(path_to_weights, self.name_model_weights)
            netG = self.load_weights_model(model_obj=netG, dir_temp_weight=dir_temp_weight, dir_model_weight=dir_model_weight)
        else:
            generator_init_type = self.config_param['model']['config_cyclegan'].get('generator_init_type', 'normal')
            htmisc.init_weights(netG, init_type=generator_init_type)
        return netG

    def get_discriminator(self, type):
        discriminator_choice = hmisc.get_nested(self.config_param, ['model', 'config_gan', f'discriminator_{type}_choice'])
        config_model = hmisc.get_nested(self.config_param, ['model', f'config_{discriminator_choice}'])
        netD = self.get_model(config_model=config_model, model_choice=discriminator_choice)

        reload_weights_discriminator_config = self.config_param['model']['config_cyclegan'].get(f'reload_discriminator_{type}_config', {})
        discriminator_gradients = self.config_param['model']['config_cyclegan'].get(f'discriminator_{type}_gradients', True)
        self.set_requires_grad([netD], discriminator_gradients)
        if reload_weights_discriminator_config.get('status', None):
            path_to_weights = reload_weights_discriminator_config['model_path']
            dir_temp_weight = os.path.join(path_to_weights, self.name_temp_weights)
            dir_model_weight = os.path.join(path_to_weights, self.name_model_weights)
            netD = self.load_weights_model(model_obj=netD, dir_temp_weight=dir_temp_weight, dir_model_weight=dir_model_weight)
        else:
            discriminator_init_type = self.config_param['model']['config_cyclegan'].get('discriminator_init_type', 'normal')
            htmisc.init_weights(netD, init_type=discriminator_init_type)

        return netD

    @abc.abstractmethod
    def get_data_generators(self, indicator_train):
        # This one may vary over implementations
        raise NotImplementedError

    def train_model(self):

        min_epoch_loss = 9999
        breakdown_counter = 0

        discriminator_loss_curve = []
        generator_loss_curve = []
        identity_loss_curve = []
        cycle_loss_curve = []
        epoch_loss_curve = []
        val_loss_curve = []


        return_gradients = self.config_param['model'].get('return_gradients', False)

        if return_gradients:
            self.history_dict.setdefault('overal_grad_gen_A2B', [])
            self.history_dict.setdefault('overal_param_gen_A2B', [])
            self.history_dict.setdefault('overal_grad_discr_A', [])
            self.history_dict.setdefault('overal_param_discr_A', [])

            self.history_dict.setdefault('overal_grad_gen_B2A', [])
            self.history_dict.setdefault('overal_param_gen_B2A', [])
            self.history_dict.setdefault('overal_grad_discr_B', [])
            self.history_dict.setdefault('overal_param_discr_B', [])

        self.netG_A2B.train()
        self.netG_B2A.train()
        self.netD_A.train()
        self.netD_B.train()

        n_epoch = self.config_param['model']['n_epoch']
        n_epoch_save = self.config_param['model'].get('n_epoch_save', int(0.10 * n_epoch))

        ind_weight_clip_discr = self.config_param['model']['config_cyclegan'].get('discriminator_clipweights', False)
        weight_clip_discr = self.config_param['model']['config_cyclegan'].get('discriminator_clipweights_value', 0)

        ind_weight_clip_gen = self.config_param['model']['config_cyclegan'].get('generator_clipweights', False)
        weight_clip_gen = self.config_param['model']['config_cyclegan'].get('generator_clipweights_value', 0)

        fake_A_buffer = htmisc.ReplayBuffer()
        fake_B_buffer = htmisc.ReplayBuffer()

        # Here comes the real training
        epoch = 0
        while epoch < n_epoch and breakdown_counter < self.breakdown_limit:
            try:
                print(f"Epoch {epoch + 1}/{n_epoch} ...")
                # Train
                epoch_loss = 0
                identity_loss_value = 0
                cycle_loss_value = 0
                generator_loss_value = 0
                discriminator_loss_value = 0

                for container in self.train_loader:
                    if self.config_param['data'].get('transform_resize', False):
                        self.train_loader.dataset.set_resize_parameters()

                    if self.debug:
                        print('EXEC - TRAIN: First batch loaded', self.debug)
                        print('Batch size is.. ', container['input'].shape)

                    if self.masked:
                        X = container['input']
                        y = container['target']
                        mask_input = container['mask_input']
                        mask_input = mask_input.to(self.device)
                        mask_target = container['mask_target']
                        mask_target = mask_target.to(self.device)
                    else:
                        X = container['input']
                        y = container['target']

                    X, y = X.to(self.device), y.to(self.device)
                    batch_size = X.shape[0]

                    if self.masked:
                        y = y * mask_target
                        X = X * mask_input

                    target_real = torch.Tensor(batch_size).fill_(1.0).to(self.device)
                    target_fake = torch.Tensor(batch_size).fill_(0.0).to(self.device)

                    self.set_requires_grad([self.netD_A, self.netD_B], False)

                    ###### Generators A2B and B2A ######
                    self.optimizer_G.zero_grad()

                    # Identity loss
                    # G_A2B(B) should equal B if real B is fed
                    same_B = self.netG_A2B(y)
                    if self.masked:
                        same_B = same_B * mask_target

                    loss_identity_B = self.criterion_identity(same_B, y) * self.lambda_identity(epoch)
                    # G_B2A(A) should equal A if real A is fed
                    same_A = self.netG_B2A(X)
                    if self.masked:
                        same_A = same_A * mask_input

                    loss_identity_A = self.criterion_identity(same_A, X) * self.lambda_identity(epoch)

                    identity_loss_value += loss_identity_A.item() / 2
                    identity_loss_value += loss_identity_B.item() / 2

                    # GAN loss
                    fake_B = self.netG_A2B(X)
                    if self.masked:
                        fake_B = fake_B * mask_target

                    pred_fake = self.netD_B(fake_B)
                    loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real)

                    fake_A = self.netG_B2A(y)
                    if self.masked:
                        fake_A = fake_A * mask_input

                    pred_fake = self.netD_A(fake_A)
                    loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real)

                    generator_loss_value += loss_GAN_A2B.item() / 2
                    generator_loss_value += loss_GAN_B2A.item() / 2

                    # Cycle loss
                    recovered_A = self.netG_B2A(fake_B)
                    if self.masked:
                        recovered_A = recovered_A * mask_input


                    loss_cycle_ABA = self.criterion_cycle(recovered_A, X) * self.lambda_cycle(epoch)

                    recovered_B = self.netG_A2B(fake_A)
                    if self.masked:
                        recovered_B = recovered_B * mask_target
                    loss_cycle_BAB = self.criterion_cycle(recovered_B, y) * self.lambda_cycle(epoch)

                    cycle_loss_value += loss_cycle_ABA.item() / 2
                    cycle_loss_value += loss_cycle_BAB.item() / 2

                    # Total loss
                    loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                    loss_G.backward()

                    self.optimizer_G.step()
                    if ind_weight_clip_gen:
                        for p in self.netG_A2B.parameters():
                            p.data.clamp_(-weight_clip_gen, weight_clip_gen)


                        for p in self.netG_B2A.parameters():
                            p.data.clamp_(-weight_clip_gen, weight_clip_gen)

                    ###################################
                    self.set_requires_grad([self.netD_A, self.netD_B], True)

                    ###### Discriminator A ######
                    self.optimizer_D_A.zero_grad()

                    # Real loss
                    pred_real = self.netD_A(X)
                    loss_D_real = self.criterion_GAN(pred_real, target_real)

                    # Fake loss
                    fake_A = fake_A_buffer.push_and_pop(fake_A)
                    pred_fake = self.netD_A(fake_A.detach())
                    loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

                    # Total loss
                    loss_D_A = (loss_D_real + loss_D_fake) * 0.5
                    loss_D_A.backward()

                    discriminator_loss_value += loss_D_A.item() / 2

                    self.optimizer_D_A.step()
                    # Clip the weight of discriminator A...
                    if ind_weight_clip_discr:
                        for p in self.netD_A.parameters():
                            p.data.clamp_(-weight_clip_discr, weight_clip_discr)
                    ###################################

                    ###### Discriminator B ######
                    self.optimizer_D_B.zero_grad()

                    # Real loss
                    pred_real = self.netD_B(y)
                    loss_D_real = self.criterion_GAN(pred_real, target_real)

                    # Fake loss
                    fake_B = fake_B_buffer.push_and_pop(fake_B)
                    pred_fake = self.netD_B(fake_B.detach())
                    loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

                    # Total loss
                    loss_D_B = (loss_D_real + loss_D_fake) * 0.5
                    loss_D_B.backward()

                    discriminator_loss_value += loss_D_B.item() / 2

                    self.optimizer_D_B.step()
                    # Clip the weight of optimizer B
                    if ind_weight_clip_discr:
                        for p in self.netD_B.parameters():
                            p.data.clamp_(-weight_clip_discr, weight_clip_discr)
                    ###################################

                    if torch.isnan(loss_D_B + loss_D_A) or torch.isinf(loss_D_B + loss_D_A):
                        break

                    epoch_loss += loss_G.item()

                    """
                    Store gradient stuff
                    """

                    # Store the gradients...
                    if return_gradients:
                        # Do it in a key/value fashion
                        networks_to_analyse = [self.netG_A2B, self.netG_B2A, self.netD_A, self.netD_B]
                        networks_grad_key = ['overal_grad_gen_A2B', 'overal_grad_gen_B2A', 'overal_grad_discr_A', 'overal_grad_discr_B']
                        networks_param_key = ['overal_param_gen_A2B', 'overal_param_gen_B2A', 'overal_param_discr_A', 'overal_param_discr_B']
                        for i, i_network in enumerate(networks_to_analyse):
                            i_param_key = networks_param_key[i]
                            i_grad_key = networks_grad_key[i]
                            list_children_network = htmisc.get_all_children(i_network, [])
                            grad_per_layer, param_per_layer = self.get_param_gradients(list_children_network)
                            self.history_dict[i_grad_key].append(grad_per_layer)
                            self.history_dict[i_param_key].append(param_per_layer)

                n_examples = self.train_loader.__len__()
                epoch_loss = epoch_loss / n_examples
                identity_loss_value = identity_loss_value / n_examples
                cycle_loss_value = cycle_loss_value / n_examples
                generator_loss_value = generator_loss_value / n_examples
                discriminator_loss_value = discriminator_loss_value / n_examples

                epoch_loss_curve.append(epoch_loss)
                identity_loss_curve.append(identity_loss_value)
                cycle_loss_curve.append(cycle_loss_value)
                generator_loss_curve.append(generator_loss_value)
                discriminator_loss_curve.append(discriminator_loss_value)

                val_loss_value = self.validate_model()
                val_loss_curve.append(val_loss_value)

                # Update learning rates
                self.lr_scheduler_G.step()
                self.lr_scheduler_D_A.step()
                self.lr_scheduler_D_B.step()

                print('loss over batch: ', epoch_loss)

                # We moeten iets....
                if not np.isfinite(epoch_loss):
                    break

                if epoch_loss < min_epoch_loss:
                    if self.debug:
                        print('INFO - EXEC: \t Old minimum loss ', min_epoch_loss)
                        print('INFO - EXEC: \t New minimum loss ', epoch_loss)
                    min_epoch_loss = epoch_loss
                    self.temp_weights = {'state_A2B': self.netG_A2B.state_dict(),
                                         'state_B2A': self.netG_B2A.state_dict(),
                                         'state_DA': self.netD_A.state_dict(),
                                         'state_DB': self.netD_B.state_dict()}

                # We want to have a decrease in loss...
                if epoch_loss != 0:
                    # For all the non-zero losses...
                    temp_curve = [x for x in val_loss_curve if x != 0]
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

                # prev_loss = epoch_loss
                epoch += 1

                # Every now and then.. save intermediate results..
                if epoch % max(n_epoch_save, 1) == 0:
                    dir_temp_weights = os.path.join(self.config_param['dir']['doutput'], self.name_temp_weights)
                    torch.save(self.temp_weights, dir_temp_weights)
                    plt.close('all')
                    self.save_model(plot_name='intermediate')

            except KeyboardInterrupt:
                break

        if breakdown_counter > self.breakdown_limit:
            print('INFO - EXEC: \t We are not advancing fast enough. Broke out of training loop')
        elif epoch >= n_epoch:
            print('INFO - EXEC: \t Completed all epochs')
        else:
            print('INFO - EXEC: \t Increase in loss.. break down')

        self.history_dict['train_loss'] = epoch_loss_curve
        self.history_dict['identity_loss'] = identity_loss_curve
        self.history_dict['cyclic_loss'] = cycle_loss_curve
        self.history_dict['generator_loss'] = generator_loss_curve
        self.history_dict['discriminator_loss'] = discriminator_loss_curve
        self.history_dict['val_loss'] = val_loss_curve

        return self.history_dict

    def test_model(self):

        # Set model's test mode
        self.netG_A2B.eval()
        self.netG_B2A.eval()

        running_loss = []
        with torch.no_grad():  # IMPORTANT
            for container in self.test_loader:
                if self.masked:
                    X = container['input']
                    y = container['target']
                    mask_input = container['mask_input']
                    mask_input = mask_input.to(self.device)
                    mask_target = container['mask_target']
                    mask_target = mask_target.to(self.device)
                else:
                    X = container['input']
                    y = container['target']

                X, y = X.to(self.device), y.to(self.device)

                fake_B = 0.5 * (self.netG_A2B(X).data + 1.0)
                fake_A = 0.5 * (self.netG_B2A(y).data + 1.0)

                if self.masked:
                    fake_A = fake_A * mask_input
                    fake_B = fake_B * mask_target

                temp_item = (self.criterion_GAN(fake_B, y).item() +
                             self.criterion_GAN(fake_A, X).item()) / 2
                running_loss.append(temp_item)

        self.test_loss = np.mean(running_loss)
        self.history_dict['test_loss'] = self.test_loss
        return running_loss

    def validate_model(self):

        self.netG_A2B.eval()
        self.netG_B2A.eval()

        loss = 0
        with torch.no_grad():  # IMPORTANT
            for container in self.val_loader:
                if self.masked:
                    X = container['input']
                    y = container['target']
                    mask_input = container['mask_input']
                    mask_input = mask_input.to(self.device)
                    mask_target = container['mask_target']
                    mask_target = mask_target.to(self.device)
                else:
                    X = container['input']
                    y = container['target']

                X = X.to(self.device)
                y = y.to(self.device)

                y_pred = self.netG_A2B(X)
                x_pred = self.netG_B2A(y)

                if self.masked:
                    y_pred = mask_target * y_pred
                    x_pred = mask_input * x_pred

                loss += (self.criterion_GAN(y_pred, y).item() + self.criterion_GAN(x_pred, X).item()) / 2.

        self.netG_A2B.train()
        self.netG_B2A.train()

        return loss / len(self.val_loader)

    @staticmethod
    def save_weight_state_dict(x, path, name):

        for prefix_name, state_dict in x.items():
            print('Storing weights for ', prefix_name)
            torch.save(state_dict, os.path.join(path, prefix_name + '_' + name))

    def save_model(self, plot_name='final'):
        doutput = self.config_param['dir']['doutput']
        dir_history = os.path.join(self.config_param['dir']['doutput'], self.name_model_hist)

        model_weights = {'state_A2B': self.netG_A2B.state_dict(),
                         'state_B2A': self.netG_B2A.state_dict(),
                         'state_DA': self.netD_A.state_dict(),
                         'state_DB': self.netD_B.state_dict()}

        self.save_weight_state_dict(self.temp_weights, doutput, self.name_temp_weights)
        self.save_weight_state_dict(model_weights, doutput, self.name_model_weights)

        # Save history object
        serialized_json_history_obj = json.dumps(self.history_dict)
        with open(dir_history, 'w') as f:
            f.write(serialized_json_history_obj)

        x0, y0, y_pred0, plot_augmentation = self.get_image_prediction()
        print('CycleGAN: Output of image prediction ')
        print('input ', x0.shape)
        print('pred ', y0.shape)
        print('target ', y_pred0.shape)

        main_path = self.config_param['dir']['doutput']

        # This is a really dirty trick to fix this issue...
        # Issue: stacking with shapes like (8, 256, 256) and (1, 256, 256) is not possible
        # But my function also accepts lists
        try:
            plot_array = np.stack([x0, y0, y_pred0], axis=0)
            print('Shape of plot array', plot_array.shape)
        except:
            plot_array = [x0, y0, y_pred0]
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

        # # Plotting of individual components of the loss....
        fig_handle_deux, ax = plt.subplots()
        id_loss = self.history_dict['identity_loss']
        cyclic_loss = self.history_dict['cyclic_loss']
        gen_loss = self.history_dict['generator_loss']
        discr_loss = self.history_dict['discriminator_loss']
        loss_list = [id_loss, cyclic_loss, gen_loss, discr_loss]
        loss_names = ['identity loss', 'cyclic loss', 'generator loss', 'discriminator loss']
        for i, i_plot in enumerate(loss_list):
            ax.plot(i_plot, label=loss_names[i])

        plt.legend()
        fig_handle.savefig(dir_history_graph)
        fig_handle_deux.savefig(dir_history_graph_comp)

        return fig_handle

    def save_weight_history(self):
        # Nothing gets returned for now...
        if self.config_param['model']['return_gradients']:
            networks_grad_key = ['overal_grad_gen_A2B', 'overal_grad_gen_B2A', 'overal_grad_discr_A',
                                 'overal_grad_discr_B']
            networks_param_key = ['overal_param_gen_A2B', 'overal_param_gen_B2A', 'overal_param_discr_A',
                                  'overal_param_discr_B']
            for i, i_network in enumerate(range(len(networks_grad_key))):
                # Fix gradients
                i_grad_key = networks_grad_key[i]
                temp_grad = self.history_dict[i_grad_key]
                gradient_fig = self.get_weight_figure(temp_grad)
                gradient_fig.suptitle(f'{i_grad_key} gradient graph')
                dir_gradient_graph = os.path.join(self.config_param['dir']['doutput'],
                                                  f'{i_grad_key}' + self.name_gradient_graph)
                gradient_fig.savefig(dir_gradient_graph)

                # Fix params..
                i_param_key = networks_param_key[i]
                temp_param = self.history_dict[i_param_key]
                param_fig = self.get_weight_figure(temp_param)
                param_fig.suptitle(f'{i_param_key} parameter graph')
                dir_param_graph = os.path.join(self.config_param['dir']['doutput'], f'{i_param_key}' + self.name_parameter_graph)
                param_fig.savefig(dir_param_graph)
        else:
            gradient_fig = plt.figure()
            param_fig = plt.figure()
            if self.debug:
                print('No gradients are calculated.')

    def load_weights(self, dir_temp_weight=None, dir_model_weight=None):
        # temp_names = [self.name_B2A, self.name_A2B, self.name_D_A, self.name_D_B]
        temp_names = [self.name_B2A, self.name_A2B]  # Actually only need two...
        doutput = self.config_param['dir']['doutput']
        dir_temp_weight_dict = {x: os.path.join(doutput, x + '_' + self.name_temp_weights) for x in temp_names}
        dir_model_weight_dict = {x: os.path.join(doutput, x + '_' + self.name_model_weights) for x in temp_names}

        bool_all_temp_weights = all([os.path.isfile(v) for k, v in dir_temp_weight_dict.items()])
        bool_all_model_weights = all([os.path.isfile(v) for k, v in dir_model_weight_dict.items()])

        hmisc.print_dict(dir_temp_weight_dict)

        if bool_all_temp_weights:
            print('Loading temp weights')
            weightB2A = torch.load(dir_temp_weight_dict[self.name_B2A], map_location=self.device)
            weightA2B = torch.load(dir_temp_weight_dict[self.name_A2B], map_location=self.device)
            # weightDA = torch.load(dir_temp_weight_dict[self.name_D_A], map_location=self.device)
            # weightDB = torch.load(dir_temp_weight_dict[self.name_D_B], map_location=self.device)
            self.netG_B2A.load_state_dict(weightB2A)
            self.netG_A2B.load_state_dict(weightA2B)
            self.netG_B2A = self.netG_B2A.float()
            self.netG_A2B = self.netG_A2B.float()

        elif bool_all_model_weights:
            print('Loading model weights')
            weightB2A = torch.load(dir_model_weight_dict[self.name_B2A], map_location=self.device)
            weightA2B = torch.load(dir_model_weight_dict[self.name_A2B], map_location=self.device)
            # weightDA = torch.load(dir_temp_weight_dict[self.name_D_A], map_location=self.device)
            # weightDB = torch.load(dir_temp_weight_dict[self.name_D_B], map_location=self.device)
            self.netG_B2A.load_state_dict(weightB2A)
            self.netG_A2B.load_state_dict(weightA2B)
            self.netG_B2A = self.netG_B2A.float()
            self.netG_A2B = self.netG_A2B.float()
        else:
            print('ERROR - EXEC: \t Please, there is no weight file yet')
            print('\t searched ', doutput)

    def get_image_prediction(self, sel_batch=None, sel_item=None, **kwargs):

        # Predicts an image using the test set. and eval() mode.
        # Afterwards post-processes the images.
        n_batches = self.test_loader.__len__()
        n_item = self.test_loader.batch_size
        print('INFO - EXEC: \t test batch size: ', n_item)
        if sel_batch is None:
            sel_batch = np.random.randint(n_batches)
        if sel_item is None:
            sel_item = np.random.randint(n_item)

        # Not doing anything with batch or item number...
        batch_counter = 0
        self.netG_A2B.eval()
        self.netG_B2A.eval()

        with torch.no_grad():  # IMPORTANT
            for container in self.test_loader:
                if self.masked:
                    X = container['input']
                    y = container['target']
                    mask_input = container['mask_input']
                    mask_input = mask_input.to(self.device)
                    mask_target = container['mask_target']
                    mask_target = mask_target.to(self.device)
                else:
                    X = container['input']
                    y = container['target']

                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.netG_A2B(X)

                if self.masked:
                    y_pred = y_pred * mask_target
                    y = y * mask_target

                if sel_batch == batch_counter:
                    break

                batch_counter += 1

        self.netG_A2B.train()
        self.netG_B2A.train()
        print('Before postproc output')
        print('X.shape', X.shape)
        print('y.shape', y.shape)
        print('y_pred.shape', y_pred.shape)
        X, y, y_pred, plot_augm = self.postproc_output(X, y, y_pred, sel_item=sel_item)
        return X, y, y_pred, plot_augm
