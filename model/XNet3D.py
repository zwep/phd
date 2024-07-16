# encoding: utf-8

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


class XNetDown3D(torch.nn.Module):
    """
    Down scale part of a UNET
    """

    def __init__(self, in_chans=1, chans=2, num_pool_layers=3, drop_prob=0.1, **kwargs):
        super().__init__()

        self.debug = kwargs.get('debug')
        device = kwargs.get('device', 'cpu')
        down_block_activation_name = kwargs.get('down_block_activation', 'relu')
        down_block_name = kwargs.get('down_block', 'ConvBlock3D')
        down_block_kernel_size = kwargs.get('down_block_kernel_size', (2, 3, 3))
        down_block_normalization_name = kwargs.get('down_block_normalization', 'BatchNorm3D')
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
                                                                  kernel_size=down_block_kernel_size,
                                                                  debug=self.debug)])
        ch = chans
        for i in range(num_pool_layers - 1):
            if self.debug:
                print(f' Xnet Down channels - {ch}, {ch * 2}')
            self.down_sample_layers += [down_block(in_chans=ch, out_chans=ch * 2, drop_prob=drop_prob,
                                                   block_activation=down_block_activation_name,
                                                   kernel_size=down_block_kernel_size,
                                                   block_normalization=down_block_normalization_name)]
            ch *= 2

        self.conv = down_block(in_chans=ch, out_chans=ch, drop_prob=drop_prob,
                               block_activation=down_block_activation_name,
                               kernel_size=down_block_kernel_size,
                               block_normalization=down_block_normalization_name)

        n_down = len(self.down_sample_layers)
        if pool_name == 'max':
            print('XNET DOWN - using max pooling')
            self.pool = [torch.nn.MaxPool3d(kernel_size=(1, 2, 2)).to(device) for _ in range(n_down)]
        elif pool_name == 'avg':
            print('XNET DOWN - using avg pooling')
            self.pool = [torch.nn.AvgPool3d(kernel_size=(1, 2, 2)).to(device) for _ in range(n_down)]
        elif pool_name == 'conv':
            self.pool = []
            ch = chans
            for _ in range(n_down):
                temp_layer = torch.nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=(1, 2, 2), stride=2).to(device)
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


class XNetFeature3D(torch.nn.Module):
    """
    Used to combine the output of multiple Down scaled Unet models
    """
    def __init__(self, n_concat, n_hidden, feature_activation=None, **kwargs):
        # Could change it here to a Sequential thing instead of a Module
        super().__init__()
        self.layer_list = torch.nn.ModuleList()  # Used to hold many modules at once..
        self.debug = kwargs.get('debug')
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


class XNetUp3D(torch.nn.Module):
    """
    Upscaled version of the UNET
    """

    def __init__(self, in_chans=1, out_chans=1, num_pool_layers=3, drop_prob=0.1, output_activation='identity', **kwargs):
        super().__init__()

        # Not sure about this one...
        self.dense_seq = torch.nn.Sequential(hlayers.SwapAxes2D(to_channel_last=True),
                                             torch.nn.Linear(out_chans, out_chans),
                                             hlayers.SwapAxes2D(to_channel_last=False))
        self.dense_seq[1].weight = torch.nn.Parameter(torch.as_tensor(np.eye(2)), requires_grad=True)
        self.dense_seq[1] = self.dense_seq[1].float()

        self.debug = kwargs.get('debug')
        self.output_activation = output_activation
        up_block_activation_name = kwargs.get('up_block_activation', 'relu')
        up_block_name = kwargs.get('up_block', 'ConvBlock3D')
        up_block_kernel_size = kwargs.get('up_block_kernel_size', (2, 3, 3))
        up_block_normalization_name = kwargs.get('up_block_normalization', 'BatchNorm3D')

        up_block = block_selector(up_block_name)

        ch = in_chans // 2  # I have no idea why this guy is here... but w/e
        self.up_sample_layers = torch.nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [up_block(in_chans=ch * 2, out_chans=ch // 2, drop_prob=drop_prob,
                                               block_activation=up_block_activation_name,
                                               block_normalization=up_block_normalization_name,
                                               kernel_size=up_block_kernel_size,
                                               debug=self.debug)]
            ch //= 2
        self.up_sample_layers += [up_block(in_chans=ch * 2, out_chans=ch, drop_prob=drop_prob,
                                           block_activation=up_block_activation_name,
                                           kernel_size=up_block_kernel_size,
                                           block_normalization=up_block_normalization_name)]

        # self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(ch, ch // 2, kernel_size=1),
        #                                  torch.nn.Conv2d(ch // 2, out_chans, kernel_size=1),
        #                                  torch.nn.Conv2d(out_chans, out_chans, kernel_size=1))
        self.conv2 = torch.nn.Sequential(torch.nn.Conv3d(ch, out_chans, kernel_size=1))
        self.output_actv_layer = activation_selector(self.output_activation)
        self.debug_display_counter = 0

    def forward(self, input, stack):
        # Apply up-sampling layers
        stack = list(stack)
        output = input

        for i, layer in enumerate(self.up_sample_layers):
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)

            if self.debug and self.debug_display_counter == 0:
                print('XNET UP - layer {} - {}'.format(i, len(self.up_sample_layers)))
                print('\t  down sample layer {}'.format(downsample_layer.shape))
                print('\t interp layer size {}'.format(layer_size))
                print('\t output size {}'.format(output.shape))
                print('\t layer {}'.format(layer))

        output = self.conv2(output)
        output = self.dense_seq(output)
        if self.debug and self.debug_display_counter == 0:
            print(f'XNET UP - layer final conv {self.conv2}')
            print('\t output size {}'.format(output.shape))

        if self.output_activation is not None:
            output = self.output_actv_layer(output)

        self.debug_display_counter += 1
        return output


class XNet3D(torch.nn.Module):
    """
    Compsite model of the ones defined above...
    """
    def __init__(self, n_pool_layers=3, start_chan=4, out_chans=2, **kwargs):
        super().__init__()
        device = kwargs.get('device', 'cpu')
        print(f'XNet Received device {device}')
        self.debug = kwargs.get('debug', False)
        n_concat = int((start_chan / 2) * 2 ** n_pool_layers)

        n_hidden = kwargs.get('n_hidden', None)
        if n_hidden is None:
            n_hidden = 2 * 8
        feature_activation = kwargs.get('feature_activation', 'identity')

        if self.debug:
            print('Size of concatenation', n_concat)

        self.dense_seq = torch.nn.Sequential(hlayers.SwapAxes3D_special(to_channel_last=True),
                                             torch.nn.Linear(2, 2),
                                             hlayers.SwapAxes3D_special(to_channel_last=False))
        self.dense_seq[1].weight = torch.nn.Parameter(torch.as_tensor(np.eye(2)), requires_grad=True)

        self.mod_down = XNetDown3D(in_chans=1, chans=start_chan, num_pool_layers=n_pool_layers, **kwargs)
        self.mod_mid = XNetFeature3D(n_hidden=n_hidden, n_concat=8, feature_activation=feature_activation)
        self.mod_up = XNetUp3D(in_chans=2*n_concat, out_chans=out_chans, num_pool_layers=n_pool_layers, **kwargs)

        self.n_concat = n_concat

    def forward(self, x):
        if self.debug:
            print('USING DEBG')
            # Split all the images apart...
        input_tensor_list = torch.split(x, split_size_or_sections=1, dim=1)
        # Apply the 2x2 weight matrix to each coil image
        input_tensor_list = [self.dense_seq(x) for i, x in enumerate(input_tensor_list)]
        # Apply the down-model (sharing weights) on each image
        output_stack_mod_down = [self.mod_down(x) for x in input_tensor_list]
        # Get the results and intermediate results as well
        result_down, stack_mod_down = zip(*output_stack_mod_down)
        print('XNET3D output results down (single)', result_down[0].shape)

        # Stack results on last (new) axis
        cat_result_down_perm = torch.stack(result_down, dim=-1)
        result_mid_perm = self.mod_mid(cat_result_down_perm)
        result_mid_split = result_mid_perm.split(1, dim=-1)
        # take the last dimension as result of split
        result_mid_split = [np.take(x, 0, axis=-1) for x in result_mid_split]
        # Calculate the up model...
        result_up = [self.mod_up(x, stack=y) for x, y in zip(result_mid_split, stack_mod_down)]
        # Stack everything...
        output = torch.cat(result_up, dim=1)
        # Now reshape into 16, y, x..?
        # That is not handy with the target of course..
        return output


if __name__ == "__main__":

    import data_generator.Rx2Tx as dg_rxtx
    from torch.utils.data import DataLoader
    import numpy as np
    import importlib
    import helper.misc as hmisc
    import torch

    ddata = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels'
    DG = dg_rxtx.DataSetSurvey2B1_all(ddata, input_shape=(8, 2, 512, 256), transform_type='complex',
                                      concatenate_complex=False, stack_ax=1)
    n_files = len(DG)
    batch_size = 5
    batch_size = hmisc.correct_batch_size(batch_size, n_files)
    print('batch size', batch_size)
    data_loader = torch.utils.data.DataLoader(DG, batch_size=batch_size)
    a, b = DG.__getitem__(0)
    a.shape

    # # # Test X NET
    X_net_model = XNet3D(start_chan=1, n_pool_layers=2, out_chans=1, debug=True).float()
    X_net_model.mod_down(a[np.newaxis, 0:1])
    X_net_model(a[np.newaxis])

    input_tensor_list = torch.split(a[np.newaxis], split_size_or_sections=1, dim=1)
    # TODO change kernel size to (1, 3, 3)..
    # But how usefull is this whole approach then...?
    input_tensor_list = [X_net_model.dense_seq(x) for i, x in enumerate(input_tensor_list)]
    input_tensor_list[0].shape
    test_conv_layer = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 3, 3))
    test_conv_layer(input_tensor_list[0]).shape
    X_net_model.mod_down.down_sample_layers[0](input_tensor_list[0])
    output_stack_mod_down = [X_net_model.mod_down(x) for x in input_tensor_list]
    # import model.UNet
    # X_net_model = model.UNet.UnetModel(in_chans=16, out_chans=16, chans=4, num_pool_layers=3, drop_prob=0.1)

    total_param = X_net_model.parameters()
    optim_obj = torch.optim.SGD(params=total_param, lr=0.002)
    dl_iter = data_loader.__iter__()

    import helper_torch.misc as htmic
    import re

    plt.close('all')

    # Visualization of the result
    hplotf.plot_3d_list(pred_model.detach().numpy()[0:1], title='result')
    hplotf.plot_3d_list(target_tensor.detach().numpy()[0:1], title='target')
    hplotf.plot_3d_list(input_tensor.detach().numpy()[0:1], title='input')


    # Get the gradients....?