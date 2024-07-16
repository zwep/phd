# encoding: utf-8

import copy
import numpy as np
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
from helper_torch.misc import activation_selector, block_selector

import torch
import torch.nn
import torch.nn.functional as F
import torch.utils
import helper.misc as hmisc
from model.Blocks import ConvBlock2D
import helper_torch.layers as hlayers
import helper_torch.misc as htmisc
import torch.nn as nn


class XNetDown(torch.nn.Module):
    """
    Down scale part of a UNET
    """

    def __init__(self, in_chans=1, chans=2, num_pool_layers=3, drop_prob=0.1, **kwargs):
        super().__init__()

        self.debug = kwargs.get('debug')
        device = kwargs.get('device', 'cpu')
        down_block_activation_name = kwargs.get('down_block_activation', 'relu')
        down_block_name = kwargs.get('down_block', 'ConvBlock2D')
        down_block_normalization_name = kwargs.get('down_block_normalization', 'BatchNorm2D')
        down_block = block_selector(down_block_name)
        pool_name = kwargs.get('down_pool', 'max')
        groups = kwargs.get('down_groups', 1)
        self.in_chans = in_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = torch.nn.ModuleList([down_block(in_chans=in_chans, out_chans=chans, drop_prob=drop_prob,
                                                                  block_activation=down_block_activation_name,
                                                                  block_normalization=down_block_normalization_name,
                                                                  groups=groups,
                                                                  debug=self.debug)])
        ch = chans
        for i in range(num_pool_layers - 1):
            if self.debug:
                print(f' Xnet Down channels - {ch}, {ch * 2}')
            self.down_sample_layers += [down_block(in_chans=ch, out_chans=ch * 2, drop_prob=drop_prob,
                                                   block_activation=down_block_activation_name,
                                                   block_normalization=down_block_normalization_name)]
            ch *= 2

        self.conv = down_block(in_chans=ch, out_chans=ch, drop_prob=drop_prob,
                               block_activation=down_block_activation_name,
                               block_normalization=down_block_normalization_name)

        n_down = len(self.down_sample_layers)
        if pool_name == 'max':
            print('XNET DOWN - using max pooling')
            self.pool = [torch.nn.MaxPool2d(kernel_size=2).to(device) for _ in range(n_down)]
        elif pool_name == 'avg':
            print('XNET DOWN - using avg pooling')
            self.pool = [torch.nn.AvgPool2d(kernel_size=2).to(device) for _ in range(n_down)]
        elif pool_name == 'conv':
            self.pool = []
            ch = chans
            for _ in range(n_down):
                temp_layer = torch.nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=2, stride=2).to(device)
                ch *= 2
                self.pool.append(temp_layer)
        else:
            print('Unkown pooling name ', pool_name)

        self.debug_display_counter = 0

    def forward(self, input):
        # Stack is used to to use as skip connection in the UNET
        stack = []
        output = input
        debug_text = []

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            if self.debug and self.debug_display_counter == 0:
                print('XNET DOWN - before pooling')
                print(f'\t pool shape {self.pool}')
                print(f'\t output shape {output.shape}')
            output = self.pool[i](output)

            if self.debug and self.debug_display_counter == 0:
                print('XNET DOWN - layer {} - {}'.format(i, len(self.down_sample_layers)))
                print('\t  down sample layer {}'.format(layer))
                print('\t output size {}'.format(output.shape))
                print('\t layer {}'.format(layer))

        output = self.conv(output)

        if self.debug and self.debug_display_counter == 0:
            print('\n'.join(debug_text))

        self.debug_display_counter += 1
        return output, stack


class XNetFeature(torch.nn.Module):
    """
    Used to combine the output of multiple Down scaled Unet models
    """
    def __init__(self, n_concat, n_hidden=None, feature_activation=None, **kwargs):
        # Could change it here to a Sequential thing instead of a Module
        super().__init__()
        self.layer_list = torch.nn.ModuleList()  # Used to hold many modules at once..
        self.debug = kwargs.get('debug')

        if n_hidden is None:
            n_hidden = n_concat

        x1 = torch.nn.Linear(in_features=n_concat, out_features=n_hidden)
        x2 = torch.nn.Linear(in_features=n_hidden, out_features=n_concat)

        if feature_activation is not None:
            x1_actv = activation_selector(feature_activation)
            x2_actv = activation_selector(feature_activation)
            self.layer_list.append(x1)
            self.layer_list.append(x1_actv)
            self.layer_list.append(x2)
            self.layer_list.append(x2_actv)
        else:
            self.layer_list.append(x1)
            self.layer_list.append(x2)

        self.debug_display_counter = 0

    def forward(self, x):
        x_out = x
        for i_layer in self.layer_list:
            x_out = i_layer(x_out)
            if self.debug and self.debug_display_counter == 0:
                print('XNET FTR ')
                print(f'\t layer {i_layer}')
                print(f'\t  output shape {x_out.shape}')

        self.debug_display_counter += 1
        return x_out


class ResNetFeature(torch.nn.Module):
    """
    Used to combine the output of multiple Down scaled Unet models
    """
    def __init__(self, in_chans, n_blocks=9, padding_type='reflect',
                 norm_layer=torch.nn.InstanceNorm2d, use_dropout=True, use_bias=True, **kwargs):
        # Could change it here to a Sequential thing instead of a Module
        super().__init__()
        self.layer_list = []
        self.debug = kwargs.get('debug')
        self.debug_display_counter = 0

        block_name = kwargs.get('block_name', 'resnetblock')
        block = htmisc.block_selector(block_name)

        for i in range(n_blocks):  # add ResNet blocks
            self.layer_list += [block(in_chans=in_chans, padding_type=padding_type,
                                      norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.layer_list = torch.nn.ModuleList(self.layer_list)

    def forward(self, x):
        for i_layer in self.layer_list:
            x = x + i_layer(x)
            if self.debug and self.debug_display_counter == 0:
                print('XNET FTR Resnet')
                print(f'\t layer {i_layer}')
                print(f'\t  output shape {x.shape}')

        self.debug_display_counter += 1
        return x


class XNetUp(torch.nn.Module):
    """
    Upscaled version of the UNET
    """

    def __init__(self, in_chans=1, out_chans=1, num_pool_layers=3, drop_prob=0.1, output_activation='identity', **kwargs):
        super().__init__()

        self.dense_seq = torch.nn.Sequential(hlayers.SwapAxes2D(to_channel_last=True),
                                             torch.nn.Linear(out_chans, out_chans),
                                             hlayers.SwapAxes2D(to_channel_last=False))
        self.dense_seq[1].weight = torch.nn.Parameter(torch.as_tensor(np.eye(2)), requires_grad=True)
        self.dense_seq[1] = self.dense_seq[1].float()
        self.debug = kwargs.get('debug')
        self.output_activation = output_activation
        up_block_activation_name = kwargs.get('up_block_activation', 'relu')
        up_block_name = kwargs.get('up_block', 'ConvBlock2D')
        up_block_normalization_name = kwargs.get('up_block_normalization', 'BatchNorm2D')

        up_block = block_selector(up_block_name)

        # ch = in_chans // 2  # I have no idea why this guy is here... but w/e
        ch = in_chans  # I have no idea why this guy is here... but w/e
        self.up_sample_layers = torch.nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [up_block(in_chans=ch, out_chans=ch // 2, drop_prob=drop_prob,
                                               block_activation=up_block_activation_name,
                                               block_normalization=up_block_normalization_name,
                                               debug=self.debug)]
            ch //= 2

        self.up_sample_layers += [up_block(in_chans=ch, out_chans=ch, drop_prob=drop_prob,
                                           block_activation=up_block_activation_name,
                                           block_normalization=up_block_normalization_name)]

        # self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(ch, ch // 2, kernel_size=1),
        #                                  torch.nn.Conv2d(ch // 2, out_chans, kernel_size=1),
        #                                  torch.nn.Conv2d(out_chans, out_chans, kernel_size=1))
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(ch, out_chans, kernel_size=1))
        self.output_actv_layer = activation_selector(self.output_activation)
        self.debug_display_counter = 0

    def forward(self, input, stack):
        # Apply up-sampling layers
        stack = list(stack)
        output = input

        for i, layer in enumerate(self.up_sample_layers):
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            if self.debug and self.debug_display_counter == 0:
                print('Xnet Up ', i)
                print('Xnet Up layer', layer)
                print('Xnet Up output shape', output.shape)

            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            if self.debug and self.debug_display_counter == 0:
                print('Xnet Up output interpolated shape', output.shape)
                print('Xnet Up downsample_layer shape', downsample_layer.shape)

            # output = torch.cat([output, downsample_layer], dim=1)
            output = (output + downsample_layer) / 2
            output = layer(output)

            if self.debug and self.debug_display_counter == 0:
                print('XNET UP - layer {} - {}'.format(i, len(self.up_sample_layers)))
                print('\t  down sample layer {}'.format(downsample_layer.shape))
                print('\t interp layer size {}'.format(layer_size))
                print('\t output size {}'.format(output.shape))
                print('\t layer {}'.format(layer))

        output = self.conv2(output)
        # output = self.dense_seq(output)
        if self.debug and self.debug_display_counter == 0:
            print(f'XNET UP - layer final conv {self.conv2}')
            print('\t output size {}'.format(output.shape))

        if self.output_activation is not None:
            output = self.output_actv_layer(output)

        self.debug_display_counter += 1
        return output


class XNet(torch.nn.Module):
    """
    Model that down samples on splitted input, combines that in a high level feature layer
    and does upsampling on separate tracks as well.
    """
    def __init__(self, n_pool_layers=3, start_chan=4, out_chans=2, **kwargs):
        super().__init__()
        device = kwargs.get('device', 'cpu')
        print(f'XNet Received device {device}')
        self.debug = kwargs.get('debug', False)
        n_concat = int((start_chan / 2) * 2 ** n_pool_layers)

        n_hidden = kwargs.get('n_hidden', 16)
        feature_activation = kwargs.get('feature_activation', 'identity')

        if self.debug:
            print('Size of concatenation', n_concat)

        self.dense_seq = torch.nn.Sequential(hlayers.SwapAxes2D(to_channel_last=True),
                                             torch.nn.Linear(2, 2),
                                             hlayers.SwapAxes2D(to_channel_last=False))
        self.dense_seq[1].weight = torch.nn.Parameter(torch.as_tensor(np.eye(2)), requires_grad=True)
        # self.dense_seq = [torch.nn.Sequential(hlayers.SwapAxes2D(to_channel_last=True),
        #                                      torch.nn.Linear(2, 2),
        #                                      torch.nn.Linear(2, 2),
        #                                      hlayers.SwapAxes2D(to_channel_last=False)) for _ in range(8)]
        # for i in range(8):
        #     self.dense_seq[i][1].weight = torch.nn.Parameter(torch.as_tensor(np.eye(2)), requires_grad=True)
        #     self.dense_seq[i][2].weight = torch.nn.Parameter(torch.as_tensor(np.eye(2)), requires_grad=True)
        #     self.dense_seq[i][1].float()
        #     self.dense_seq[i][2].float()
        # self.dense_seq = [x.to(device) for x in self.dense_seq]

        self.mod_down = XNetDown(in_chans=2, chans=start_chan, num_pool_layers=n_pool_layers, **kwargs)
        self.mod_mid = XNetFeature(n_hidden=n_hidden, n_concat=8, feature_activation=feature_activation)
        self.mod_up = XNetUp(in_chans=n_concat, out_chans=out_chans, num_pool_layers=n_pool_layers, **kwargs)

        self.n_concat = n_concat

    def forward(self, x):
        if self.debug:
            print('USING DEBG')
        # Version 2.1 - Dense layer for everyone!
        input_tensor_list = torch.split(x, 2, dim=1)
        # input_tensor_list = [self.dense_seq[i](x) for i, x in enumerate(input_tensor_list)]  # Model...
        input_tensor_list = [self.dense_seq(x) for i, x in enumerate(input_tensor_list)]  # Model...

        output_stack_mod_down = [self.mod_down(x) for x in input_tensor_list]  # Model...
        result_down, stack_mod_down = zip(*output_stack_mod_down)

        # Version 1.1 - No concatting, but stacking. Different concat layer as well
        cat_result_down_perm = torch.stack(result_down, dim=-1)
        result_mid_perm = self.mod_mid(cat_result_down_perm)
        result_mid_split = result_mid_perm.split(1, dim=-1)
        result_mid_split = [x[:, :, :, :, 0] for x in result_mid_split]  # take the last dimension as result of split

        result_up = [self.mod_up(x, stack=y) for x, y in zip(result_mid_split, stack_mod_down)]  # Model...
        output = torch.cat(result_up, dim=1)

        return output


class XNetZeroDawn(torch.nn.Module):
    """Differs from Xnet by some Resnet blocks..."""
    def __init__(self, in_chan, out_chan, start_chan, n_layer, n_blocks, n_split, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)

        conv_layer_name = kwargs.get('conv_layer', 'conv2d')
        block_name = kwargs.get('block_name', 'resnetblock')
        mid_block = htmisc.block_selector(block_name)

        down_block_normalization_name = kwargs.get('down_block_normalization', 'InstanceNorm2D')
        down_activation_name = kwargs.get('down_activation', 'relu')

        feature_activation = kwargs.get('feature_activation', 'identity')

        up_block_normalization_name = kwargs.get('up_block_normalization', 'InstanceNorm2D')
        up_activation_name = kwargs.get('up_activation', 'relu')

        final_activation_name = kwargs.get('final_activation', 'tanh')
        drop_prob = kwargs.get('drop_prob', 0.1)

        self.n_rotation_layer = kwargs.get('n_rotation_layer', 0)

        self.n_split = n_split

        # This is not a block per se....
        # self.pre_model = torch.nn.ModuleList([hlayers.LearnableRotation() for _ in range(n_split)])
        # self.pre_model = hlayers.RotationLayer(N=self.n_rotation_layer)

        # # # One model
        down_sample_model = [nn.ReflectionPad2d(3),
                 htmisc.module_selector(conv_layer_name)(in_chan, start_chan, kernel_size=7, padding=0),
                 htmisc.module_selector(down_block_normalization_name)(start_chan),
                 nn.ReLU(True)]

        for i in range(n_layer):  # add downsampling layers
            mult = 2 ** i
            down_sample_model += [htmisc.module_selector(conv_layer_name)(start_chan * mult, start_chan * mult * 2, kernel_size=3, stride=2, padding=1),
                        htmisc.module_selector(down_block_normalization_name)(start_chan * mult * 2),
                        htmisc.activation_selector(down_activation_name)]
        self.down_sample_model = nn.Sequential(*down_sample_model)
        # # # One model

        # # # Second model
        # I could add a Dense Layer... that acts on chan, ny, nx, 8
        # Before I even act out this...

        resnet_blocks_1 = []
        mult = 2 ** n_layer
        for i in range(n_blocks//2):  # add ResNet blocks
            norm_layer = htmisc.module_selector(down_block_normalization_name)
            resnet_blocks_1 += [mid_block(in_chans=start_chan * mult, norm_layer=norm_layer, drop_prob=drop_prob)]

        self.resnet_blocks_1 = nn.Sequential(*resnet_blocks_1)

        # # # Intermezo model
        n_concat = 8 * start_chan * mult
        # n_hidden = n_concat // 2
        # n_hidden = n_concat
        n_hidden = 2 * n_concat
        x1 = torch.nn.Linear(in_features=n_concat, out_features=n_hidden)
        x1_actv = activation_selector(feature_activation)
        x2 = torch.nn.Linear(in_features=n_hidden, out_features=n_concat)
        x2_actv = activation_selector(feature_activation)
        concat_model = [x1, x1_actv, x2, x2_actv]
        self.concat_model = nn.Sequential(*concat_model)

        self.swap_ax_to_last = hlayers.SwapAxes2D(to_channel_last=True)
        self.swap_ax_to_first = hlayers.SwapAxes2D(to_channel_last=False)

        # # # And back to Resnet
        resnet_blocks_2 = []
        mult = 2 ** n_layer
        for i in range(n_blocks//2, n_blocks):  # add ResNet blocks
            norm_layer = htmisc.module_selector(down_block_normalization_name)
            resnet_blocks_2 += [mid_block(in_chans=start_chan * mult, norm_layer=norm_layer, drop_prob=drop_prob)]

        self.resnet_blocks_2 = nn.Sequential(*resnet_blocks_2)
        # # # Second model

        # # # Third model
        up_sample_model = []
        for i in range(n_layer):  # add upsampling layers
            mult = 2 ** (n_layer - i)
            norm_layer = htmisc.module_selector(up_block_normalization_name)
            # Might wish to use an interpolation method instead...
            # output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            up_sample_model += [nn.ConvTranspose2d(start_chan * mult, int(start_chan * mult / 2),
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1),
                        norm_layer(int(start_chan * mult / 2)),
                        htmisc.activation_selector(up_activation_name)]

        up_sample_model += [nn.ReflectionPad2d(3)]
        up_sample_model += [htmisc.module_selector(conv_layer_name)(start_chan, out_chan, kernel_size=7, padding=0)]
        final_activation = htmisc.activation_selector(final_activation_name)
        up_sample_model += [final_activation]
        # # # Third model
        # self.up_sample_model = nn.ModuleList([nn.Sequential(*copy.deepcopy(up_sample_model)) for _ in range(self.n_split)])
        self.up_sample_model = nn.ModuleList([nn.Sequential(*up_sample_model) for _ in range(self.n_split)])
        self.debug_display_counter = 0

    def forward(self, input):
        """Standard forward"""

        input_tensor_list = torch.chunk(input, self.n_split, dim=1)

        ## HEre we have a setup for Learnable ROtation
        # temp_premodel = []
        # for i, x in enumerate(input_tensor_list):
        #     x_temp = self.swap_ax_to_last(x)
        #     x_temp = self.pre_model[i](x_temp)
        #     x_temp = self.swap_ax_to_first(x_temp)
        #     temp_premodel.append(x_temp)
        #     # input_tensor_list[i] = x_temp

        ## Here we have a setup for ROtation Layer (no learning
        # temp_premodel = []
        # for i, x in enumerate(input_tensor_list):
        #     x_temp = self.swap_ax_to_last(x)
        #     x_temp = self.pre_model(x_temp)
        #     ny, nx = x_temp.shape[1:3]
        #     x_temp = x_temp.reshape(-1, ny, nx, 2 * self.n_rotation_layer)
        #     x_temp = self.swap_ax_to_first(x_temp)
        #     temp_premodel.append(x_temp)
            # input_tensor_list[i] = x_temp

        # input_tensor_list = temp_premodel

        input_tensor_list = [self.down_sample_model(x) for i, x in enumerate(input_tensor_list)]
        if self.debug and self.debug_display_counter == 0:
            print('XNET zerodawn - after first model')
            print('length input...', len(input_tensor_list))
            print('shape input...', input_tensor_list[0].shape)

        first_resnet_output = [self.resnet_blocks_1(x) for x in input_tensor_list]
        if self.debug and self.debug_display_counter == 0:
            print('XNET zerodawn - after first resnet')
            print('length input...', len(first_resnet_output))
            print('shape input...', first_resnet_output[0].shape)

        # second_resnet_output = [self.resnet_blocks_2(x) for x in first_resnet_output]
        # concat_resnet = torch.cat(second_resnet_output, dim=1)

        concat_resnet = torch.cat(first_resnet_output, dim=1)

        if self.debug and self.debug_display_counter == 0:
            print('XNET zerodawn - after first concat')
            print('shape...', concat_resnet.shape)
        concat_resnet = self.swap_ax_to_last(concat_resnet)
        if self.debug and self.debug_display_counter == 0:
            print('XNET zerodawn - swap to last')
            print('shape...', concat_resnet.shape)
        dense_output = self.concat_model(concat_resnet)
        if self.debug and self.debug_display_counter == 0:
            print('XNET zerodawn - dense net out')
            print('shape...', dense_output.shape)
        dense_output = self.swap_ax_to_first(dense_output)
        if self.debug and self.debug_display_counter == 0:
            print('XNET zerodawn - swap to first')
            print('shape...', dense_output.shape)
        chunked_output = torch.chunk(dense_output, chunks=8, dim=1)
        if self.debug and self.debug_display_counter == 0:
            print('XNET zerodawn - after chunk')
            print('length input...', len(chunked_output))
            print('shape input...', chunked_output[0].shape)

        second_resnet_output = [self.resnet_blocks_2(x) for x in chunked_output]
        up_output = [self.up_sample_model[i](x) for i, x in enumerate(second_resnet_output)]

        # up_output = [self.model_up[i](x) for i, x in enumerate(chunked_output)]
        if self.debug and self.debug_display_counter == 0:
            print('XNET zerodawn - after model up')
            print('length input...', len(up_output))
            print('shape input...', up_output[0].shape)

        output = torch.cat(up_output, dim=1)

        self.debug_display_counter += 1
        return output


if __name__ == "__main__":

    import data_generator.Rx2Tx as dg_rxtx
    from torch.utils.data import DataLoader
    import numpy as np
    import importlib
    import helper.misc as hmisc
    import torch

    ddata = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels'
    DG = dg_rxtx.DataSetSurvey2B1_all(ddata, input_shape=(16, 512, 256), transform_type='complex')
    n_files = len(DG)
    batch_size = 5
    batch_size = hmisc.correct_batch_size(batch_size, n_files)
    print('batch size', batch_size)
    data_loader = torch.utils.data.DataLoader(DG, batch_size=batch_size)
    a, b = DG.__getitem__(0)




    # # # Test X NET
    X_net_model = XNet(start_chan=2, n_pool_layers=3, out_chans=2, down_block='convblock2d').float()
    z = X_net_model(a[np.newaxis, :])

    # import model.UNet
    # X_net_model = model.UNet.UnetModel(in_chans=16, out_chans=16, chans=4, num_pool_layers=3, drop_prob=0.1)

    total_param = X_net_model.parameters()
    optim_obj = torch.optim.SGD(params=total_param, lr=0.002)
    dl_iter = data_loader.__iter__()

    import helper_torch.misc as htmic
    import re

    plt.close('all')

    # # ## Test XResnet Feautre
    model_obj = XNetZeroDawn(in_chan=2, out_chan=2, start_chan=2, n_layer=2, n_blocks=2, n_split=8, debug=True)
    a = torch.as_tensor(np.random.rand(16, 256, 256)).float()
    res = model_obj.forward(a[np.newaxis])
    hplotf.plot_3d_list(res.detach().numpy())

    import helper.misc as hmisc
    import json
    derp = '{"name": "unet_survey2b1", "comment": "-purpose of this run-", "packed_keys": [], "dir": {"ddata": "/data/seb/flavio_npy", "doutput": null, "dtemplate": null}, "model": {"n_epoch": 150, "config_gan": {"conditional": true, "generator_choice": "xnetzerodawn", "discriminator_choice": "nlayersplit", "generator_loss": "PerceptualLossStyleLoss", "discriminator_clipweights": true, "gan_mode": "lsgan", "smoothed_target": true, "n_discriminator_training": 2, "lambda_l1": 6, "lr_generator": 0.0001, "lr_discriminator": 0.0004, "config_xnetzerodawn": {"conv_layer": "conv2d", "in_chan": 2, "out_chan": 2, "start_chan": 2, "n_layer": 4, "n_blocks": 8, "n_split": 8, "final_activation": "identity"}, "config_nlayersplit": {"input_nc": 4, "n_layers": 6, "ndf": 64, "n_split": 8, "normalization_layer": "EvoNorm2D"}}, "loss": "L1Loss", "config": {"input_shape": [16, 256, 256]}, "init_type": "orthogonal", "return_gradients": true, "model_choice": "gan", "xccn_lambda": 0.001}, "optimizer": {"name": "Adam", "config": {"lr": 0.001}, "policy_config": {"base_lr": 0.001, "max_lr": 0.005, "step_size_up": 500}, "policy": "linear"}, "data": {"batch_perc": 0.02, "complex_type": "cartesian", "masked": true, "num_workers": 0, "input_is_output": false, "fourier_transform": false, "transform_type": "complex", "batch_size": 2, "relative_phase": false, "generator_choice": "multiple_flavio", "random_phase": true}, "gpu_frac": 0.9, "callback": {"breakdown_limit": 20, "memory_length": 5, "memory_time": 15}}'
    res = json.loads(derp)
    hmisc.print_dict(res)