# encoding: utf-8

"""
Here we define a generic object ot run model stuff.
"""


# Standard packages
import abc
import os
import inspect
import sys
import json
import numpy as np
import functools
import GPUtil
import logging
import warnings

if '__file__' in vars():
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_path)

# Self created code
import helper.nvidia_parser as hnvidia
import helper_torch.loss as hloss
import helper_torch.loss_physics as hloss_physics
import torchgeometry.losses as geom_loss

import helper.plot_class as hplotc
import helper.misc as hmisc
import helper.array_transf as harray
import helper_torch.misc as htmisc
import helper_torch.loss_weighting as hloss_weight

# Torch code
import torch
import torch.nn
import torch.utils.data
import torch.nn.modules.loss as tloss
import torch.optim as toptimizer
import helper_torch.optimizer as hoptimizer
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

# Load models

import model.DeepDenseNet as model_deepdense
import model.UNet as model_unet
import model.VGG as model_vgg
import model.XNet as model_xnet
import model.XXNet as model_xxnet
import model.XXXNet as model_xxxnet
import model.GroupedXNet as model_xnet_group
import model.BridgeXNet as model_xnet_bridge
import model.NlayerNet as model_nlayernet
import model.ResNet as model_resnet
import model.YNet as model_ynet
import model.UNetSkipConnection as model_unetskip
import model.Discriminator as model_discriminator
import model.ESPCN as model_espcn
import model.AE as model_ae
import model.ShimNet as model_shimnet
import torchvision.models as torchvision_models


class ExecutorBase:
    def __init__(self, model_path=None, config_file=None, **kwargs):
        assert (config_file is not None) or (model_path is not None), 'Nothing is supplied'

        if config_file is not None and model_path is None:
            self.config_param = config_file
            self.model_path = self.config_param['dir']['doutput']
        else:
            self.model_path = model_path
            self.name_config_param = kwargs.get('name_config_param', 'config_param.json')
            self.config_param = self.load_config(name=self.name_config_param)

        self.debug = kwargs.get('debug', False)
        self.debug_cuda = kwargs.get('debug_cuda', False)
        self.inference = kwargs.get('inference', False)
        self.load_model_only = kwargs.get('load_model_only', False)

        if self.debug:
            print("Executor Base \t DEBUG MODE IS ACTIVATED")
            if self.load_model_only:
                print("Executor Base \t ONLY LOADING MODEL. No data generators. No loss objects")
            if self.inference:
                print("Executor Base \t INFERENCE MODE. Only test generators.")

        self.name_temp_weights = 'temp_weights.pt'
        self.name_model_weights = 'model_weights.pt'
        self.name_model_hist = 'model_history.json'
        self.name_loss_graph = 'losses_graph.jpg'
        self.name_loss_component_graph = 'losses_component_graph.jpg'
        self.name_gradient_graph = 'gradient_graph.jpg'
        self.name_parameter_graph = 'parameter_graph.jpg'

        self.name_test_input = 'test_input'
        self.name_test_pred = 'test_pred'
        self.name_test_output = 'test_output'
        self.name_test_diff = 'test_diff'
        self.name_test_fig = 'test_fig'

        # Store configuration parameters in the object
        # This section is for data transformations
        # shows which complex type is used (polar/cartesian)
        self.complex_type = self.config_param['data'].get('complex_type', None)
        # transformation on input and output
        # If I ever forget it...
        self.transform_type = self.config_param['data'].get('transform_type', 'real')
        # transformatin only on output
        self.transform_type_target = self.config_param['data'].get('transform_type_target', None)
        if self.transform_type_target is None:
            self.transform_type_target = self.transform_type

        # if we have fouriertransformed the input...
        self.fourier_transform = self.config_param['data'].get('fourier_transform', False)
        # Could potentially hold data transformation tasks
        self.transform_obj = None

        # Define possible lambda-loss-weighting functions
        self.loss_weight_dict = {}
        hloss_weight_class = dict(inspect.getmembers(hloss_weight, inspect.isclass))
        self.loss_weight_dict.update(hloss_weight_class)

        # Define possible losses
        self.loss_dict = {}
        loss_member_class = dict(inspect.getmembers(tloss, inspect.isclass))
        hloss_member_fun = dict(inspect.getmembers(hloss, inspect.isfunction))
        hloss_member_class = dict(inspect.getmembers(hloss, inspect.isclass))
        hloss_physics_member_class = dict(inspect.getmembers(hloss_physics, inspect.isclass))
        geom_torch_class = dict(inspect.getmembers(geom_loss, inspect.isclass))
        self.loss_dict.update(loss_member_class)
        self.loss_dict.update(hloss_member_fun)
        self.loss_dict.update(hloss_member_class)
        self.loss_dict.update(hloss_physics_member_class)
        self.loss_dict.update(geom_torch_class)

        # Define possible optimizers
        self.optimizer_dict = {}
        optimizer_member = dict(inspect.getmembers(toptimizer, inspect.isclass))
        optimizer_helper = dict(inspect.getmembers(hoptimizer, inspect.isclass))
        self.optimizer_dict.update(optimizer_member)
        self.optimizer_dict.update(optimizer_helper)

        # Set device
        index_gpu = kwargs.get('index_gpu', None)
        device = kwargs.get('device', None)
        if index_gpu is None:
            index_gpu, p_gpu = hnvidia.get_free_gpu_id()

        if device is None:
            self.device = torch.device("cuda:{}".format(index_gpu) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # These should be filled during the train procedure..
        self.history_dict = {}
        self.history_dict.setdefault('train_loss', [])
        self.history_dict.setdefault('val_loss', [])
        self.history_dict.setdefault('test_loss', [])

        # Set tolerance on improvement
        self.breakdown_limit = self.config_param['callback'].get('breakdown_limit', 0)
        self.memory_length = self.config_param['callback'].get('memory_length', 5)
        self.memory_time = self.config_param['callback'].get('memory_time', 5)

        # Paramters during training...
        self.masked = self.config_param['data'].get('masked', False)

        # Needs to be set during training (chosen as set of weights with minimal loss)
        self.temp_weights = {}

        # Needs to be set by running test_model()
        self.test_loss = None

        # Create model - initialized by child classes
        self.model_obj = None

        # Create optimizer - initialized by child classes
        self.optimizer_obj = None

        # This one needs to be defined in the specific objective class if you want to use it. (in __init__ of subclass)
        self.trained_model_config = None
        # This one needs to be defined in the specific objective class if you want to use it. (in __init__)
        self.trained_modelrun_obj = None

        # Maybe post here a whole info file about the current status of this object..?

    def load_config(self, name='config_param.json'):
        # Load the right parameters...
        with open(os.path.join(self.model_path, name), 'r') as f:
            temp = f.read()
            config_param = json.loads(temp)

        return config_param

    def load_weights_model(self, model_obj, dir_temp_weight=None, dir_model_weight=None, device=None):
        if device is None:
            device = self.device

        if os.path.isfile(dir_temp_weight):
            print('Loading temp weights')
            # We prefer the temp_weights file over the model_weights.. This is for now only true in the keras config
            weight_dict = torch.load(dir_temp_weight, map_location=device)
            model_obj.load_state_dict(weight_dict)
            model_obj = model_obj.float()
        elif os.path.isfile(dir_model_weight):
            print('Loading model weights')
            weight_dict = torch.load(dir_model_weight, map_location=device)
            model_obj.load_state_dict(weight_dict)
            model_obj = model_obj.float()
        else:
            htmisc.init_weights(model_obj)
            warnings.warn(f'ERROR - EXEC: \t Please, there is no weight file yet. Initializing with normal\n '
                          f'\t searched {dir_temp_weight}, {dir_model_weight}')

        return model_obj

    def get_model_config(self, nested_key_path):
        model_choice = hmisc.get_nested(self.config_param, nested_key_path)
        key_name = 'config_' + model_choice
        config_model = self.config_param['model'].get(key_name, None)
        # Backward compatibility with older config files...
        if config_model is None:
            print(f"We have not found model choice {key_name}. Defaulting to config_gan")
            config_gan = self.config_param['model']['config_gan']
            if config_gan is not None:
                config_model = config_gan.get(key_name, None)

        if self.debug:
            print('Config model...')
            hmisc.print_dict(config_model)

        return config_model, model_choice

    def get_model(self, config_model, model_choice, device=None):
        if device is None:
            device = self.device

        if model_choice == 'generator':
            print('Deprecated generator.')
            # n_pool_layer_gen = int(config_generator['n_pool_layers'])
            # embedding_shape = (int(ny / 2 ** n_pool_layer_gen), int(nx / 2 ** n_pool_layer_gen))
            # print('Executor ACGAN - embedding shape - ', embedding_shape)
            # model_obj = Generator.GeneratorConditionalGan(debug=self.debug, embedding_shape=embedding_shape, **config_generator)
            # generator.index_tensor = generator.index_tensor.to(self.device)

        elif model_choice == 'espcn':
            model_obj = model_espcn.ESPCN(debug=self.debug, **config_model)

        elif model_choice == 'espcn_norm':
            model_obj = model_espcn.ESPCN_normalisation(debug=self.debug, **config_model)

        elif model_choice == 'espcn_rev':
            model_obj = model_espcn.ESPCN_reversed(debug=self.debug, **config_model)

        elif model_choice == 'espcn_deconv':
            model_obj = model_espcn.ESPCN_deconv(debug=self.debug, **config_model)

        elif model_choice == 'espcn_xconv':
            model_obj = model_espcn.ESPCN_xconv(debug=self.debug, **config_model)

        elif model_choice == 'xnet':
            model_obj = model_xnet.XNet(device=self.device, **config_model)

        elif model_choice == 'unet_model':
            model_obj = model_unet.UnetModel(device=self.device, debug=self.debug, **config_model)

        elif model_choice == 'unet':
            model_obj = model_unet.UNet(device=self.device, debug=self.debug, **config_model)

        elif model_choice == 'xnetzerodawn':
            model_obj = model_xnet.XNetZeroDawn(device=self.device, **config_model)

        elif model_choice == 'unetskip':
            norm_layer_name = config_model.get('normalization_layer', 'InstanceNorm2d')
            norm_layer = htmisc.module_selector(module_name=norm_layer_name)
            if 'instance' in norm_layer_name.lower():
                norm_layer = functools.partial(norm_layer, affine=False, track_running_stats=False)

            model_obj = model_unetskip.UnetGenerator(norm_layer=norm_layer, debug=self.debug, **config_model)

        elif model_choice == 'resnet':
            # Get norm layer....
            norm_layer_name = config_model.get('normalization_layer', 'InstanceNorm2d')  # InstanceNorm2d
            norm_layer = htmisc.module_selector(module_name=norm_layer_name)
            if 'instance' in norm_layer_name.lower():
                norm_layer = functools.partial(norm_layer, affine=False, track_running_stats=False)
            model_obj = model_resnet.ResnetGenerator(norm_layer=norm_layer, **config_model)

        elif model_choice == 'resnet50':
            model_obj = torchvision_models.resnet50()

        elif model_choice == 'resnetmsg':
            # Get norm layer....
            norm_layer_name = config_model.get('normalization_layer', 'InstanceNorm2d')  # InstanceNorm2d
            norm_layer = htmisc.module_selector(module_name=norm_layer_name)
            if 'instance' in norm_layer_name.lower():
                norm_layer = functools.partial(norm_layer, affine=False, track_running_stats=False)
            model_obj = model_resnet.ResnetGeneratorMSG(norm_layer=norm_layer, **config_model)

        elif model_choice == 'resnetsplit':
            # Get norm layer....
            norm_layer_name = config_model.get('normalization_layer', 'InstanceNorm2d')  # InstanceNorm2d
            norm_layer = htmisc.module_selector(module_name=norm_layer_name)
            if 'instance' in norm_layer_name.lower():
                norm_layer = functools.partial(norm_layer, affine=False, track_running_stats=False)
            model_obj = model_resnet.ResnetSplit(norm_layer=norm_layer, **config_model)

        elif model_choice == 'ynet':
            model_obj = model_ynet.YNet(debug=self.debug, **config_model)
        elif model_choice == 'coolynet':
            model_obj = model_ynet.TheCoolYNet(debug=self.debug, **config_model)

        elif model_choice == 'discriminator':
            ny, nx = self.config_param['data']['input_shape'][-2:]
            n_pool_layer_discr = int(config_model['n_pool_layers'])
            start_ch = int(config_model['start_ch'])
            concat_ch = start_ch * 2 ** n_pool_layer_discr
            concat_ny = ny / 4 ** n_pool_layer_discr
            n_features = int(concat_ch * concat_ny ** 2)

            model_obj = model_discriminator.Discriminator(debug=self.debug, n_features=n_features, **config_model)

        elif model_choice == 'nlayer':
            norm_layer_name = config_model.get('normalization_layer', 'InstanceNorm2d')  # InstanceNorm2d
            norm_layer = htmisc.module_selector(module_name=norm_layer_name)
            if 'instance' in norm_layer_name.lower():
                norm_layer = functools.partial(norm_layer, affine=False, track_running_stats=False)

            model_obj = model_discriminator.NLayerDiscriminator(norm_layer=norm_layer, **config_model)

        elif model_choice == 'pixel':
            norm_layer_name = config_model.get('normalization_layer', 'InstanceNorm2d')  # InstanceNorm2d
            norm_layer = htmisc.module_selector(module_name=norm_layer_name)
            if 'instance' in norm_layer_name.lower():
                norm_layer = functools.partial(norm_layer, affine=False, track_running_stats=False)
            model_obj = model_discriminator.PixelDiscriminator(norm_layer=norm_layer, debug=self.debug, **config_model)

        elif model_choice == 'pixelsplit':
            norm_layer_name = config_model.get('normalization_layer', 'InstanceNorm2d')  # InstanceNorm2d
            norm_layer = htmisc.module_selector(module_name=norm_layer_name)
            if 'instance' in norm_layer_name.lower():
                norm_layer = functools.partial(norm_layer, affine=False, track_running_stats=False)

            model_obj = model_discriminator.PixelSplitDiscriminator(norm_layer=norm_layer, **config_model)

        elif model_choice == 'deeppixel':
            norm_layer_name = config_model.get('normalization_layer', 'InstanceNorm2d')  # InstanceNorm2d
            norm_layer = htmisc.module_selector(module_name=norm_layer_name)
            if 'instance' in norm_layer_name.lower():
                norm_layer = functools.partial(norm_layer, affine=False, track_running_stats=False)

            model_obj = model_discriminator.DeepPixelDiscriminator(norm_layer=norm_layer, **config_model)

        elif model_choice == 'nlayermsg':
            norm_layer_name = config_model.get('normalization_layer', 'InstanceNorm2d')  # InstanceNorm2d
            norm_layer = htmisc.module_selector(module_name=norm_layer_name)
            if 'instance' in norm_layer_name.lower():
                norm_layer = functools.partial(norm_layer, affine=False, track_running_stats=False)

            model_obj = model_discriminator.NLayerDiscriminatorMSG(norm_layer=norm_layer, **config_model)

        elif model_choice == 'nlayersplit':
            norm_layer_name = config_model.get('normalization_layer', 'InstanceNorm2d')  # InstanceNorm2d
            norm_layer = htmisc.module_selector(module_name=norm_layer_name)
            if 'instance' in norm_layer_name.lower():
                norm_layer = functools.partial(norm_layer, affine=False, track_running_stats=False)

            model_obj = model_discriminator.NLayerSplitDiscriminator(norm_layer=norm_layer, **config_model)
        elif model_choice == 'ae':
            model_obj = model_ae.AE(**config_model)
        elif model_choice == 'nlayernet':
            model_obj = model_nlayernet.NLayerNet(**config_model)
        elif model_choice == 'shimnet':
            model_obj = model_shimnet.ShimNet(**config_model)
        elif model_choice == 'piennet':
            model_obj = model_shimnet.PienNet(**config_model)
        else:
            print('No known model type asked ...', model_choice)
            model_obj = None

        model_obj = model_obj.float()
        if self.debug_cuda:
            print('\nBefore putting model on GPU')
            GPUtil.showUtilization()

        model_obj.to(device)
        if self.debug_cuda:
            print('\nAfter putting model on GPU')
            GPUtil.showUtilization()

        model_parameters = filter(lambda p: p.requires_grad, model_obj.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        print('\nChosen model: ', model_choice)
        print('Number of parameters', params, end='\n\n')

        return model_obj

    def get_lambda_weight_fun(self, config):
        lambda_type = config.get('type', None)
        if lambda_type is None:
            lambda_loss = self.loss_weight_dict.get('Constant')
            config = {'weight': 1, 'epoch_on': 0}
        else:
            lambda_loss = self.loss_weight_dict.get(lambda_type, None)

        if lambda_loss is None:
            print('Unkown lambda loss class: ', lambda_type, config)
            lambda_loss_obj = None
        else:
            lambda_loss_obj = lambda_loss(**config)

        return lambda_loss_obj

    def get_data_loader(self, indicator_train, num_workers=0):
        data_generator = self.get_data_generators(indicator_train)
        n_files = len(data_generator)

        batch_perc = self.config_param['data'].get('batch_perc', None)
        batch_size = self.config_param['data'].get('batch_size', None)
        if batch_size is None and batch_perc:
            batch_size = n_files * batch_perc
            # Correct batch size is needed for the final plotting function
            # THere we select a random item from the batch.
            # But if the batchsize is not a multiple of the total length.. we get errors
            batch_size = int(hmisc.correct_batch_size(batch_size, n_files))

        if self.debug:
            print(f'EXEC - \t ===== {indicator_train} ===== ')
            print(f'EXEC - \t Batch perc/nfiles {batch_perc}/{n_files}')
            print(f'EXEC - \t Batch size initial {batch_size}')
        # Correct batch size is needed
        batch_size = int(hmisc.correct_batch_size(batch_size, n_files))

        if self.debug:
            print(f'EXEC - \t Batch size correction {batch_size}')

        data_loader = torch.utils.data.DataLoader(data_generator,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  shuffle=data_generator.shuffle)
        return data_loader

    def get_optimizer(self, model_parameters, optimizer_config):
        optimizer_name = optimizer_config['name']
        optimizer_config = optimizer_config['config']

        optimizer_class = self.optimizer_dict.get(optimizer_name, None)
        if optimizer_class:
            optimizer_obj = optimizer_class(params=model_parameters, **optimizer_config)
        else:
            optimizer_obj = None

        return optimizer_obj

    def get_loss(self, loss_name, options=None):
        if options is None:
            options = {}

        if self.debug:
            print('Chosen the following loss ', loss_name, options)

        if loss_name is not None:
            loss_obj = self.loss_dict.get(loss_name, None)
            if inspect.isclass(loss_obj):
                if 'Perceptual' in loss_name:
                    vgg = model_vgg.Vgg16(requires_grad=False)
                    vgg.float()
                    loss_obj = loss_obj(vgg_model=vgg, **options)
                else:
                    loss_obj = loss_obj(**options)

                loss_obj.to(self.device)

            if loss_obj is None:
                print('Warning: unkown loss  ', loss_name, options)
        else:
            loss_obj = None

        return loss_obj

    @abc.abstractmethod
    def get_data_generators(self, indicator_train):
        # This one may vary over implementations
        raise NotImplementedError

    def set_requires_grad(self, network, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            network (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(network, list):
            network = [network]
        for i_network in network:
            if i_network is not None:
                for param in i_network.parameters():
                    param.requires_grad = requires_grad

    def get_lr_scheduler(self, optimizer):
        lr_policy = self.config_param['optimizer'].get('policy', None)
        policy_config = self.config_param['optimizer'].get('policy_config', {})
        n_epoch = self.config_param['model']['n_epoch']

        if lr_policy == 'linear':
            def lambda_rule(epoch=None):
                lr_l = 1.0 - max(0, epoch - n_epoch) / float(100 + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == 'cyclic':
            # Change this to the per optimizer object specified lr.
            policy_config['base_lr'] = optimizer.param_groups[0]['lr']
            scheduler = lr_scheduler.CyclicLR(optimizer, **policy_config)
        else:
            def lambda_rule(epoch=None):
                return 1
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            if self.debug:
                print(f'We have not found a proper policy: {lr_policy}')

        return scheduler

    def get_weight_figure(self, x):
        # Here we plot the figure for the weight history, being either its gradient or the param norm
        x_switch = hmisc.change_list_order(x)
        n_layers = len(x_switch)

        fig, ax = plt.subplots(1, n_layers, figsize=(20, 15))
        # If we have multiple axes... siwtch them..
        if hasattr(ax, 'ravel'):
            ax = ax.ravel()
        else:
            ax = [ax]

        temp_max_y = np.max(x_switch)

        if np.isfinite(temp_max_y):
            temp_max_y += 0.05 * temp_max_y
        else:
            temp_max_y = 1

        for i in range(n_layers):
            temp_min, temp_mean, temp_max = zip(*x_switch[i])
            ax[i].plot(temp_min, 'b', alpha=0.5)
            ax[i].plot(temp_mean, '-.k')
            ax[i].plot(temp_max, 'r', alpha=0.5)
            # USe a global maxmimum
            # ax[i].set_ylim(0, temp_max_y)
            # OR a local maximum. For now this is desired I guess...
            # Filter out the not finite elements.
            plot_max = 1.2 * np.max([x for x in temp_max if np.isfinite(x)])
            ax[i].set_ylim(0, plot_max)

        return fig

    def save_weight_history(self):
        # Here we call upon the creation of the figure of gradient/parameter norm
        if self.config_param['model']['return_gradients']:
            if self.config_param['model']['model_choice'] == 'gan':
                temp_grad = self.history_dict['overal_grad_gen']
                gradient_fig = self.get_weight_figure(temp_grad)
                gradient_fig.suptitle('Generator gradient graph')
                dir_gradient_graph = os.path.join(self.config_param['dir']['doutput'], 'gen_' + self.name_gradient_graph)
                gradient_fig.savefig(dir_gradient_graph)

                # Do the same for params...
                temp_param = self.history_dict['overal_param_gen']
                param_fig = self.get_weight_figure(temp_param)
                param_fig.suptitle('Generator parameter graph')
                dir_param_graph = os.path.join(self.config_param['dir']['doutput'], 'gen_' + self.name_parameter_graph)
                param_fig.savefig(dir_param_graph)

                temp_grad = self.history_dict['overal_grad_discr']
                gradient_fig = self.get_weight_figure(temp_grad)
                gradient_fig.suptitle('Discriminator gradient graph')
                dir_gradient_graph = os.path.join(self.config_param['dir']['doutput'], 'discr_' + self.name_gradient_graph)
                gradient_fig.savefig(dir_gradient_graph)

                # Do the same for params...
                temp_param = self.history_dict['overal_param_discr']
                param_fig = self.get_weight_figure(temp_param)
                param_fig.suptitle('Discriminator parameter graph')
                dir_param_graph = os.path.join(self.config_param['dir']['doutput'], 'discr_' + self.name_parameter_graph)
                param_fig.savefig(dir_param_graph)
            else:
                temp_grad = self.history_dict['overal_grad']
                gradient_fig = self.get_weight_figure(temp_grad)
                dir_gradient_graph = os.path.join(self.config_param['dir']['doutput'], self.name_gradient_graph)
                gradient_fig.savefig(dir_gradient_graph)

                # Do the same for params...
                temp_param = self.history_dict['overal_param']
                param_fig = self.get_weight_figure(temp_param)
                dir_param_graph = os.path.join(self.config_param['dir']['doutput'], self.name_parameter_graph)
                param_fig.savefig(dir_param_graph)
        else:
            gradient_fig = plt.figure()
            param_fig = plt.figure()
            if self.debug:
                print('No gradients are calculated.')

        return gradient_fig, param_fig

    @staticmethod
    def get_param_gradients(list_children):
        sel_layer_name, sel_layer_param = htmisc.get_all_parameters(list_children)

        grad_level = htmisc.get_grad_layers(sel_layer_param, sel_layer_name)
        param_level = htmisc.get_param_layers(sel_layer_param, sel_layer_name)
        grad_name, grad_array = zip(*grad_level)
        param_name, param_array = zip(*param_level)
        grad_per_layer = [(float(x.min()), float(x.mean()), float(x.max())) for x in grad_array]
        param_per_layer = [(float(x.min()), float(x.mean()), float(x.max())) for x in param_array]

        return param_per_layer, grad_per_layer

    @abc.abstractmethod
    def get_image_prediction(self, sel_batch=None, sel_item=None, **kwargs):
        # This one may vary over implementations
        raise NotImplementedError

    def store_test_predictions(self, sel_dataset=0):
        # Creates a new folder in the /config..
        # Get the correct target path
        # Get all the test results..
        target_path = os.path.join(self.model_path, 'test_predictions')
        if not os.path.isdir(target_path):
            print('Target path is created')
            os.makedirs(target_path)

        n_batches = self.test_loader.__len__()
        n_item = self.test_loader.batch_size
        for i_batch in range(n_batches):
            for i_item in range(n_item):
                x0, y0, y_pred0, _ = self.get_image_prediction(sel_batch=i_batch, sel_item=i_item)
                plot_array = [x0, y0, y_pred0]
                output_path = os.path.join(target_path, f"{sel_dataset}_{i_batch}_{i_item}.jpg")
                fig_handle = hplotc.ListPlot(plot_array, figsize=(15, 10), dpi=75, augm='np.real')
                fig_handle.figure.savefig(output_path, bbox_inches='tight', pad_inches=0.0)

    def postproc_output(self, torch_input, torch_target, torch_pred, sel_item):
        # Here we perform post processing on complex outputs
        # We expect the input to be of shape (batch, chan * cpx_chan ,y ,x)
        # We are going to get images that are of (batch, chan, cpx_chan, y, x)
        # So I need to fix that still..

        if self.debug:
            print('Post processing output')

        if self.device.type == "cpu":
            torch_input = torch_input.numpy()
            torch_target = torch_target.numpy()
            torch_pred = torch_pred.numpy()
        else:
            # Then we assume CUDA..
            torch_input = torch_input.cpu().numpy()
            torch_target = torch_target.cpu().numpy()
            torch_pred = torch_pred.cpu().numpy()

        if self.debug:
            print('\t Output shape from model')
            print('\t torch input ', torch_input.shape)
            print('\t torch target ', torch_target.shape)
            print('\t torch pred ', torch_pred.shape)

        real_valued_target_types = ['imag', 'abs', 'real', 'angle', 'arcsinh', 'abssumcomplex', 'sumcomplex']

        # If we have complex numbers we process according to the amount of images we have
        if (self.transform_type == 'complex') or (self.transform_type == 'cos'):
            if self.debug:
                print('\t Transform type is complex')
            plot_augmentation = ['np.abs', 'np.angle', 'np.real', 'np.imag']

            # Yeah pretty hack-y
            # Yeah this does not work well...
            if self.trained_model_config.get('status', False):
                x0 = torch_input[sel_item]
            else:
                x0 = self.postproc_complex_array(torch_input[sel_item])

            # Here we list the exceptions when the predicted target is actually real valued.
            if self.transform_type_target in real_valued_target_types:
                if self.debug:
                    print('\t Target transform type is real/imag/abs/angle')
                # Select index 0 because it outputs one channel.
                # Maybe there will be cases where we dont have that..?
                y0, y_pred = (torch_target[sel_item], torch_pred[sel_item])
            else:
                if self.debug:
                    print('\t Target transform type is also complex')

                y0 = self.postproc_complex_array(torch_target[sel_item])
                y_pred = self.postproc_complex_array(torch_pred[sel_item])

        # If we have real valued.. the procedure is the same in all cases
        else:
            x0 = torch_input[sel_item]

            if self.transform_type_target in real_valued_target_types:
                plot_augmentation = ['np.real']
                if self.debug:
                    print('\t Target transform type is real/imag/abs/angle')
                # Select index 0 because it outputs one channel.
                # Maybe there will be cases where we dont have that..?
                y0, y_pred = (torch_target[sel_item], torch_pred[sel_item])
            else:
                plot_augmentation = ['np.abs', 'np.angle', 'np.real', 'np.imag']
                if self.debug:
                    print('\t Target transform type is also complex')

                y0 = self.postproc_complex_array(torch_target[sel_item])
                y_pred = self.postproc_complex_array(torch_pred[sel_item])

        # This is only applicable when using the Inhomogeneity removal generator
        # Should be used in conjunction with transform_type_target = real
        if self.config_param['data'].get('target_type', None) == 'expansion':
            # When we enable the option 'expansion' we assume that the output represents the
            # basis in the fourier space that needs to be summed to obtain the actual prediction
            y0 = y0.sum(axis=0)
            y_pred = y_pred.sum(axis=0)
            print('We have summed stuff... new shapes: ', y0.shape, y_pred.shape)

        if self.debug:
            print('\t Transform type is complex')
            print('\t Output shape of x0 is ', x0.shape)
            print('\t Output shape of y0 is ', y0.shape)
            print('\t Output shape of y_pred is ', y_pred.shape)

        return x0, y0, y_pred, plot_augmentation

    def postproc_complex_array(self, x):
        # Since all complex arrays are split over a channel axis, we need a separate function
        # to convert them to complex values arrays again.
        if self.debug:
            print('Post process complex array')
            print('\t Input image shape ', x.shape)

        img_shape = x.shape[-2:]
        n_y, n_x = img_shape
        # Extract the amount of channels..
        # n_c, n_y, n_x = self.config_param['data']['input_shape']
        # We know that it is complex. And know the exact position of the channels..
        n_chan = x.shape[-3] // 2

        if self.debug:
            print('\t Amount of channels n_chan ', n_chan)

        moved_shape = (n_y, n_x, n_chan, 2)
        new_shape = (n_chan * 2, n_x, n_y)

        # Move complex channel to last index...
        x = np.moveaxis(x, 0, -1)
        if n_chan == 1:
            # add a new channel axis to the data
            x = harray.to_complex(x, complex_type=self.complex_type)[np.newaxis]
        if n_chan > 1:
            x = x.reshape(moved_shape).T.reshape(new_shape).T
            x = harray.to_complex_chan(x, img_shape=img_shape, n_chan=n_chan, complex_type=self.complex_type)
            # Doing this for now... could do a squeeze on complex chan thing...
            # Reason for doing this was: plotting gave an extra dimension which made
            # organizing the names more dificult
            x = x[0]

        if self.debug:
            print('\t Output of to complex ', x.shape)

        return x


class DecisionmakerBase:
    def __init__(self, model_path=None, config_file=None, config_name='config_param.json', **kwargs):
        assert (config_file is not None) or (model_path is not None), 'Nothing is supplied'

        if config_file is not None and model_path is None:
            self.config_param = config_file
            self.model_path = self.config_param['dir']['doutput']
        else:
            self.model_path = model_path
            self.config_param = self.load_config(name=config_name)

        self.kwargs = kwargs
        self.debug = kwargs.get('debug', False)
        self.debug_cuda = kwargs.get('debug_cuda', False)

    def load_config(self, name='config_param.json'):
        with open(os.path.join(self.model_path, name), 'r') as f:
            temp = f.read()
            config_param = json.loads(temp)

        return config_param

    def decision_maker(self):
        return NotImplementedError
