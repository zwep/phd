# encoding: utf-8

import helper_torch.layers as hlayers
import torch
import torch.nn as nn
import helper_torch.misc as htmisc
from helper_torch.misc import activation_selector, module_selector

# Splits multiple input into one..

class YNet(torch.nn.Module):
    def __init__(self, in_chan, out_chan, start_chan, n_layer, n_blocks, n_split, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)

        final_activation_name = kwargs.get('final_activation', 'tanh')
        conv_layer_name = kwargs.get('conv_layer', 'conv2d')
        block_name = kwargs.get('block_name', 'resnetblock')
        block = htmisc.block_selector(block_name)
        down_block_normalization_name = kwargs.get('down_block_normalization', 'InstanceNorm2D')

        self.n_split = n_split
        drop_prob = kwargs.get('drop_prob', 0.1)

        # This is not a block per se....
        # # # One model
        model_1 = [nn.ReflectionPad2d(3),
                 htmisc.module_selector(conv_layer_name)(in_chan, start_chan, kernel_size=7, padding=0),
                 htmisc.module_selector(down_block_normalization_name)(start_chan),
                 nn.ReLU(True)]

        for i in range(n_layer):  # add downsampling layers
            mult = 2 ** i
            model_1 += [htmisc.module_selector(conv_layer_name)(start_chan * mult, start_chan * mult * 2, kernel_size=3, stride=2, padding=1),
                        htmisc.module_selector(down_block_normalization_name)(start_chan * mult * 2),
                        nn.ReLU(True)]
        self.model_down = nn.Sequential(*model_1)
        # # # One model

        # # # Second model
        # I could add a Dense Layer... that acts on chan, ny, nx, 8
        # Before I even act out this...

        model_2 = []
        mult = 2 ** n_layer
        for i in range(n_blocks//2):  # add ResNet blocks
            norm_layer = htmisc.module_selector(down_block_normalization_name)
            model_2 += [block(in_chans=start_chan * mult, norm_layer=norm_layer, drop_prob=drop_prob)]

        self.model_mid_1 = nn.Sequential(*model_2)

        # # # Intermezo model
        feature_activation = kwargs.get('feature_activation', 'identity')
        n_concat = self.n_split * start_chan * mult
        n_hidden = n_concat
        x1 = torch.nn.Linear(in_features=n_concat, out_features=n_hidden)
        x1_actv = activation_selector(feature_activation)
        x2 = torch.nn.Linear(in_features=n_hidden, out_features=n_concat)
        x2_actv = activation_selector(feature_activation)
        model_3 = [x1, x1_actv, x2, x2_actv]
        self.mod_dense = nn.Sequential(*model_3)
        # Misc layers used to do proper multiplication with the Linear layers
        self.swap_ax_to_last = hlayers.SwapAxes2D(to_channel_last=True)
        self.swap_ax_to_first = hlayers.SwapAxes2D(to_channel_last=False)

        # # # And back to Resnet
        model_4 = []
        mult = 2 ** n_layer
        for i in range(n_blocks//2, n_blocks):  # add ResNet blocks
            norm_layer = htmisc.module_selector(down_block_normalization_name)
            model_4 += [block(in_chans=start_chan * mult, norm_layer=norm_layer, drop_prob=drop_prob)]

        self.model_mid_2 = nn.Sequential(*model_4)
        # # # Second model

        # # # Third model
        model_5 = []
        for i in range(n_layer):  # add upsampling layers
            mult = 2 ** (n_layer - i)
            norm_layer = htmisc.module_selector(down_block_normalization_name)
            if i == 0:
                init_chan = start_chan * mult * self.n_split
            else:
                init_chan = start_chan * mult
            model_5 += [nn.ConvTranspose2d(init_chan, int(start_chan * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(start_chan * mult / 2)),
                      nn.ReLU(True)]

        model_5 += [nn.ReflectionPad2d(3)]
        model_5 += [htmisc.module_selector(conv_layer_name)(start_chan, out_chan, kernel_size=7, padding=0)]
        final_activation = htmisc.activation_selector(final_activation_name)
        model_5 += [final_activation]
        # # # /Third model

        self.model_up = nn.Sequential(*model_5)
        self.debug_display_counter = 0
        self.debug_model_name = 'YNet'

    def forward(self, input):
        """Standard forward"""

        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} - before first model')
            print('input shape...', input.shape)
            print('Print coming model... ', self.model_down)

        input_tensor_list = torch.chunk(input, self.n_split, dim=1)
        input_tensor_list = [self.model_down(x) for i, x in enumerate(input_tensor_list)]
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} - after first model')
            print('length input...', len(input_tensor_list))
            print('shape input...', input_tensor_list[0].shape)

        first_resnet_output = [self.model_mid_1(x) for x in input_tensor_list]
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} after first resnet')
            print('length input...', len(first_resnet_output))
            print('shape input...', first_resnet_output[0].shape)

        concat_resnet = torch.cat(first_resnet_output, dim=1)

        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} after first concat')
            print('shape...', concat_resnet.shape)

        concat_resnet = self.swap_ax_to_last(concat_resnet)
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} swap to last')
            print('shape...', concat_resnet.shape)
        dense_output = self.mod_dense(concat_resnet)
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} dense net out')
            print('shape...', dense_output.shape)
        dense_output = self.swap_ax_to_first(dense_output)
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} swap to first')
            print('shape...', dense_output.shape)

        up_output = self.model_up(dense_output)
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} after chunk')
            print('length input...', len(up_output))
            print('shape input...', up_output[0].shape)

        self.debug_display_counter += 1
        return up_output


class TheCoolYNet(torch.nn.Module):
    def __init__(self, in_chan, out_chan, start_chan, n_layer, n_blocks, n_split, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)

        final_activation_name = kwargs.get('final_activation', 'tanh')
        conv_layer_name = kwargs.get('conv_layer', 'conv2d')
        block_name = kwargs.get('block_name', 'resnetblock')
        block = htmisc.block_selector(block_name)
        down_block_normalization_name = kwargs.get('down_block_normalization', 'InstanceNorm2D')

        self.n_split = n_split
        drop_prob = kwargs.get('drop_prob', 0.1)

        # This is not a block per se....
        # This is a pre-rotation model..
        self.pre_model = torch.nn.ModuleList([hlayers.LearnableRotation() for _ in range(n_split)])

        # # # One model
        model_1 = [nn.ReflectionPad2d(3),
                   htmisc.module_selector(conv_layer_name)(in_chan, start_chan, kernel_size=7, padding=0),
                   htmisc.module_selector(down_block_normalization_name)(start_chan),
                   nn.ReLU(True)]

        for i in range(n_layer):  # add downsampling layers
            mult = 2 ** i
            model_1 += [htmisc.module_selector(conv_layer_name)(start_chan * mult, start_chan * mult * 2, kernel_size=3,
                                                                stride=2, padding=1),
                        htmisc.module_selector(down_block_normalization_name)(start_chan * mult * 2),
                        nn.ReLU(True)]
        self.model_down = nn.Sequential(*model_1)
        # # # One model

        # # # Second model
        # I could add a Dense Layer... that acts on chan, ny, nx, 8
        # Before I even act out this...

        model_2 = []
        mult = 2 ** n_layer
        for i in range(n_blocks // 2):  # add ResNet blocks
            norm_layer = htmisc.module_selector(down_block_normalization_name)
            model_2 += [block(in_chans=start_chan * mult, norm_layer=norm_layer, drop_prob=drop_prob)]

        self.model_mid_1 = nn.Sequential(*model_2)

        # # # Intermezo model
        feature_activation = kwargs.get('feature_activation', 'identity')
        n_concat = 8 * start_chan * mult
        n_hidden = 2 * n_concat
        n_out = start_chan * mult ##
        x1 = torch.nn.Linear(in_features=n_concat, out_features=n_hidden)
        x1_actv = activation_selector(feature_activation)
        x2 = torch.nn.Linear(in_features=n_hidden, out_features=n_out) ##
        # x2 = torch.nn.Linear(in_features=n_hidden, out_features=n_concat)
        x2_actv = activation_selector(feature_activation)
        model_3 = [x1, x1_actv, x2, x2_actv]
        self.mod_dense = nn.Sequential(*model_3)
        # Misc layers used to do proper multiplication with the Linear layers
        self.swap_ax_to_last = hlayers.SwapAxes2D(to_channel_last=True)
        self.swap_ax_to_first = hlayers.SwapAxes2D(to_channel_last=False)

        # # # And back to Resnet
        model_4 = []
        mult = 2 ** n_layer
        for i in range(n_blocks // 2, n_blocks):  # add ResNet blocks
            norm_layer = htmisc.module_selector(down_block_normalization_name)
            model_4 += [block(in_chans=start_chan * mult, norm_layer=norm_layer, drop_prob=drop_prob)]

        # self.model_mid_2 = nn.Sequential(*model_4)
        # # # Second model

        # # # Third model
        model_5 = []
        for i in range(n_layer):  # add upsampling layers
            mult = 2 ** (n_layer - i)
            norm_layer = htmisc.module_selector(down_block_normalization_name)
            if i == 0:
                init_chan = n_out ##
                # init_chan = start_chan * mult * self.n_split
            else:
                init_chan = start_chan * mult
            model_5 += [nn.ConvTranspose2d(init_chan, int(start_chan * mult / 2),
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1),
                        norm_layer(int(start_chan * mult / 2)),
                        nn.ReLU(True)]

        model_5 += [nn.ReflectionPad2d(3)]
        model_5 += [htmisc.module_selector(conv_layer_name)(start_chan, out_chan, kernel_size=7, padding=0)]
        final_activation = htmisc.activation_selector(final_activation_name)
        model_5 += [final_activation]
        # # # /Third model

        self.model_up = nn.Sequential(*model_5)
        self.debug_display_counter = 0
        self.debug_model_name = 'TheCoolYNet'

    def forward(self, input):
        """Standard forward"""

        input_tensor_list = torch.chunk(input, self.n_split, dim=1)
        # This is added
        temp_premodel = []
        for i, x in enumerate(input_tensor_list):
            x_temp = self.swap_ax_to_last(x)
            x_temp = self.pre_model[i](x_temp)
            x_temp = self.swap_ax_to_first(x_temp)
            temp_premodel.append(x_temp)
            # input_tensor_list[i] = x_temp

        input_tensor_list = temp_premodel
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} - after pre model')
            print('length input...', len(input_tensor_list))
            print('shape input...', input_tensor_list[0].shape)

        input_tensor_list = [self.model_down(x) for i, x in enumerate(input_tensor_list)]
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} - after first model')
            print('length input...', len(input_tensor_list))
            print('shape input...', input_tensor_list[0].shape)

        first_resnet_output = [self.model_mid_1(x) for x in input_tensor_list]
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} after first resnet')
            print('length input...', len(first_resnet_output))
            print('shape input...', first_resnet_output[0].shape)

        concat_resnet = torch.cat(first_resnet_output, dim=1)
        # concat_resnet = torch.stack(first_resnet_output, dim=1)

        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} after first concat')
            print('shape...', concat_resnet.shape)

        concat_resnet = self.swap_ax_to_last(concat_resnet)
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} swap to last')
            print('shape...', concat_resnet.shape)
        dense_output = self.mod_dense(concat_resnet)
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} dense net out')
            print('shape...', dense_output.shape)
        dense_output = self.swap_ax_to_first(dense_output)
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} swap to first')
            print('shape...', dense_output.shape)

        # # This should be added to have another split thing after the concatenation
        # chunk_resnet = torch.chunk(dense_output, self.n_split, dim=1)
        # resnet_split_output = [self.model_mid_2(x) for x in chunk_resnet]
        # resnet_output = torch.cat(resnet_split_output, dim=1)
        # up_output = self.model_up(resnet_output)

        up_output = self.model_up(dense_output)
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} after chunk')
            print('length input...', len(up_output))
            print('shape input...', up_output[0].shape)

        self.debug_display_counter += 1
        return up_output


class ArbitraryYNet(torch.nn.Module):
    def __init__(self, in_chan, out_chan, start_chan, n_layer, n_blocks, n_split, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)

        final_activation_name = kwargs.get('final_activation', 'tanh')
        conv_layer_name = kwargs.get('conv_layer', 'conv2d')
        block_name = kwargs.get('block_name', 'resnetblock')
        block = htmisc.block_selector(block_name)
        down_block_normalization_name = kwargs.get('down_block_normalization', 'InstanceNorm2D')

        self.n_split = n_split
        drop_prob = kwargs.get('drop_prob', 0.1)

        # This is not a block per se....
        # # # One model
        model_1 = [nn.ReflectionPad2d(3),
                 htmisc.module_selector(conv_layer_name)(in_chan, start_chan, kernel_size=7, padding=0),
                 htmisc.module_selector(down_block_normalization_name)(start_chan),
                 nn.ReLU(True)]

        for i in range(n_layer):  # add downsampling layers
            mult = 2 ** i
            model_1 += [htmisc.module_selector(conv_layer_name)(start_chan * mult, start_chan * mult * 2, kernel_size=3, stride=2, padding=1),
                        htmisc.module_selector(down_block_normalization_name)(start_chan * mult * 2),
                        nn.ReLU(True)]
        self.model_down = nn.Sequential(*model_1)
        # # # One model

        # # # Second model
        # I could add a Dense Layer... that acts on chan, ny, nx, 8
        # Before I even act out this...

        model_2 = []
        mult = 2 ** n_layer
        for i in range(n_blocks//2):  # add ResNet blocks
            norm_layer = htmisc.module_selector(down_block_normalization_name)
            model_2 += [block(in_chans=start_chan * mult, norm_layer=norm_layer, drop_prob=drop_prob)]

        self.model_mid_1 = nn.Sequential(*model_2)

        # # # Intermezo model
        feature_activation = kwargs.get('feature_activation', 'identity')
        n_concat = self.n_split * start_chan * mult
        n_hidden = n_concat
        x1 = torch.nn.Linear(in_features=n_concat, out_features=n_hidden)
        x1_actv = activation_selector(feature_activation)
        x2 = torch.nn.Linear(in_features=n_hidden, out_features=n_concat)
        x2_actv = activation_selector(feature_activation)
        model_3 = [x1, x1_actv, x2, x2_actv]
        self.mod_dense = nn.Sequential(*model_3)
        # Misc layers used to do proper multiplication with the Linear layers
        self.swap_ax_to_last = hlayers.SwapAxes2D(to_channel_last=True)
        self.swap_ax_to_first = hlayers.SwapAxes2D(to_channel_last=False)

        # # # And back to Resnet
        model_4 = []
        mult = 2 ** n_layer
        for i in range(n_blocks//2, n_blocks):  # add ResNet blocks
            norm_layer = htmisc.module_selector(down_block_normalization_name)
            model_4 += [block(in_chans=start_chan * mult, norm_layer=norm_layer, drop_prob=drop_prob)]

        self.model_mid_2 = nn.Sequential(*model_4)
        # # # Second model

        # # # Third model
        model_5 = []
        for i in range(n_layer):  # add upsampling layers
            mult = 2 ** (n_layer - i)
            norm_layer = htmisc.module_selector(down_block_normalization_name)
            if i == 0:
                init_chan = start_chan * mult * self.n_split
            else:
                init_chan = start_chan * mult
            model_5 += [nn.ConvTranspose2d(init_chan, int(start_chan * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(start_chan * mult / 2)),
                      nn.ReLU(True)]

        model_5 += [nn.ReflectionPad2d(3)]
        model_5 += [htmisc.module_selector(conv_layer_name)(start_chan, out_chan, kernel_size=7, padding=0)]
        final_activation = htmisc.activation_selector(final_activation_name)
        model_5 += [final_activation]
        # # # /Third model

        self.model_up = nn.Sequential(*model_5)
        self.debug_display_counter = 0
        self.debug_model_name = 'YNet'

    def forward(self, input):
        """Standard forward"""

        input_tensor_list = torch.chunk(input, self.n_split, dim=1)
        input_tensor_list = [self.model_down(x) for i, x in enumerate(input_tensor_list)]
        if self.debug and self.debug_display_counter == 0:
            print('XNET zerodawn - after first model')
            print('length input...', len(input_tensor_list))
            print('shape input...', input_tensor_list[0].shape)

        first_resnet_output = [self.model_mid_1(x) for x in input_tensor_list]
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} after first resnet')
            print('length input...', len(first_resnet_output))
            print('shape input...', first_resnet_output[0].shape)

        concat_resnet = torch.cat(first_resnet_output, dim=1)

        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} after first concat')
            print('shape...', concat_resnet.shape)

        concat_resnet = self.swap_ax_to_last(concat_resnet)
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} swap to last')
            print('shape...', concat_resnet.shape)
        dense_output = self.mod_dense(concat_resnet)
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} dense net out')
            print('shape...', dense_output.shape)
        dense_output = self.swap_ax_to_first(dense_output)
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} swap to first')
            print('shape...', dense_output.shape)

        up_output = self.model_up(dense_output)
        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} after chunk')
            print('length input...', len(up_output))
            print('shape input...', up_output[0].shape)

        self.debug_display_counter += 1
        return up_output


if __name__ == "__main__":
    import numpy as np
    import helper.plot_fun as hplotf
    import helper.plot_class as hplotc
    import torch
    import data_generator.InhomogRemoval as data_gen

    dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation_rxtx'
    gen = data_gen.DataGeneratorInhomogRemoval(ddata=dir_data, dataset_type='test', complex_type='polar')
    a, b = gen.__getitem__(0)

    mod_obj = YNet(in_chan=2, out_chan=1, start_chan=2, n_layer=2, n_blocks=4, n_split=8, debug=True)
    res = mod_obj(a[np.newaxis])
    res.shape
    hplotc.SlidingPlot(res.detach().numpy())

    mod_obj = TheCoolYNet(in_chan=2, out_chan=1, start_chan=32, n_layer=4, n_blocks=4, n_split=8, debug=True)
    res = mod_obj(a[np.newaxis])
    res.shape
    hplotc.SlidingPlot(res.detach().numpy())
