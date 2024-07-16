# encoding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
import functools

import helper_torch.misc as htmisc


"""
Here we defined some Discriminators for the GAN models we use.
For now we collect them all here.

"""

import torch
import torch.nn as nn
import model.Blocks as Blocks
import helper_torch.layers as hlayer


class Discriminator(nn.Module):
    def __init__(self, start_ch=16, n_pool_layers=3, groups=8, n_features=4096, **kwargs):
        super().__init__()
        ch = start_ch
        self.debug = kwargs.get('debug')
        self.groups = groups
        down_sample = []
        print('pool layers and type ', n_pool_layers, type(n_pool_layers))
        for i_layer in range(n_pool_layers):
            temp_down = Blocks.ConvBlock2D(in_chans=ch, out_chans=2 * ch, groups=self.groups)
            temp_sample = nn.Conv2d(in_channels=2 * ch, out_channels=2 * ch, kernel_size=4, stride=4, groups=1)
            down_sample.append(temp_down)
            down_sample.append(temp_sample)
            ch = 2 * ch

        self.down_sample = nn.Sequential(*down_sample)
        self.source_layer = nn.Sequential(nn.Linear(in_features=n_features//self.groups, out_features=1), nn.Sigmoid())
        self.coil_layer = nn.Sequential(nn.Linear(in_features=n_features//self.groups, out_features=self.groups), nn.Sigmoid())

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        temp = self.down_sample(x)
        if self.debug:
            print('Discriminator down sample:', temp.shape)

        temp = temp.chunk(chunks=self.groups, dim=1)
        temp = [x.reshape((batch_size, -1)) for x in temp]
        if self.debug:
            print('Discriminator chunck + unravel')
            print([x.shape for x in temp])
        validity = [self.source_layer(x) for x in temp]
        if self.debug:
            print('Discriminator validity')
            print([x.shape for x in validity])
        coil_position = [self.coil_layer(x) for x in temp]
        if self.debug:
            print('Discriminator coil_position')
            print([x.shape for x in coil_position])

        return validity, coil_position


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    # USed this code
    # https://cvnote.ddlee.cn/2019/09/02/cyclegan-pytorch-github
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, **kwargs):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        self.debug = kwargs.get('debug')
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)
        self.debug_display_counter = 0

    def forward(self, input, **kwargs):
        """Standard forward."""
        if self.debug and self.debug_display_counter == 0:
            print('Input of Pixel Discriminator', input.shape)

        self.debug_display_counter += 1
        return self.net(input)


class DeepPixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, in_chan, n_layer=3, start_chan=32, norm_layer=nn.BatchNorm2d, **kwargs):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        self.debug = kwargs.get('debug')
        self.debug_display_counter = 0

        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation_name = kwargs.get('activation', 'leakyrelu')
        activation_config = kwargs.get('activation_config', {})
        conv_layer_name = kwargs.get('conv_layer', 'Conv2d')
        conv_layer_config = kwargs.get('conv_layer_config', {})

        self.net = [htmisc.module_selector(conv_layer_name)(in_chan, start_chan, **conv_layer_config),
                    htmisc.activation_selector(activation_name, config=activation_config),
                    htmisc.module_selector(conv_layer_name)(start_chan, start_chan, **conv_layer_config),
                    norm_layer(start_chan)]

        temp_chan = start_chan
        for i_block in range(n_layer-1):
            temp_chan = 2 * temp_chan
            temp = [htmisc.module_selector(conv_layer_name)(int(temp_chan/2), temp_chan, **conv_layer_config),
                    htmisc.activation_selector(activation_name, config=activation_config),
                    htmisc.module_selector(conv_layer_name)(temp_chan, temp_chan, **conv_layer_config),
                    norm_layer(temp_chan)]

            self.net.extend(temp)

        self.net.extend([norm_layer(temp_chan),
                         nn.Conv2d(temp_chan, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)])

        self.net = nn.ModuleList(self.net)

    def forward(self, input, **kwargs):
        """Standard forward."""
        for i_layer in self.net:
            if self.debug and self.debug_display_counter == 0:
                print('Computing layer ', i_layer)
                print('\tSize to input layer ', input.shape)

            input = i_layer(input)

        self.debug_display_counter += 1
        return input


class PixelSplitDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, n_split, ndf=64, norm_layer=nn.BatchNorm2d, **kwargs):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.dense_layer = nn.Linear(in_features=n_split, out_features=2)
        self.swap_to_last = hlayer.SwapAxes2D(to_channel_last=True)
        self.swap_to_first = hlayer.SwapAxes2D(to_channel_last=False)
        self.n_split = n_split
        self.net = nn.Sequential(*self.net)

    def forward(self, input, **kwargs):
        """Standard forward."""
        input_chunked = torch.chunk(input, self.n_split, dim=1)
        output_chunked = [self.net(x) for x in input_chunked]
        output_cat = torch.cat(output_chunked, dim=1)
        output_cat = self.swap_to_last(output_cat)
        output_cat = self.dense_layer(output_cat)
        output_cat = self.swap_to_first(output_cat)

        return output_cat


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, **kwargs):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        conv_layer_name = kwargs.get('conv_layer', 'conv2d')
        conv_layer = htmisc.module_selector(conv_layer_name)

        kw = 4
        padw = 1
        sequence = [conv_layer(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                conv_layer(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            conv_layer(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [conv_layer(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.ModuleList(sequence)

        self.debug_display_counter = 0
        self.debug = kwargs.get('debug', False)

    def forward(self, input, **kwargs):
        """Standard forward."""
        if self.debug_display_counter == 0 and self.debug:
            print('NLayerDiscriminator intermediate output')

        for i_layer in self.model:
            input = i_layer(input)
            if self.debug and self.debug_display_counter == 0:

                print(f'\t Output layer shape ', input.shape)

        self.debug_display_counter += 1
        return input


class NLayerSplitDiscriminator(nn.Module):
    """ SPlit version of the discriminator..."""
    def __init__(self, input_nc, n_split, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, **kwargs):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        self.dense_layer = nn.Linear(in_features=n_split, out_features=2 * n_split)
        self.swap_to_last = hlayer.SwapAxes2D(to_channel_last=True)
        self.swap_to_first = hlayer.SwapAxes2D(to_channel_last=False)
        self.n_split = n_split
        self.net = nn.Sequential(*sequence)

        self.debug_display_counter = 0
        self.debug = kwargs.get('debug', False)

    def forward(self, input, **kwargs):
        """Standard forward."""
        if self.debug_display_counter == 0 and self.debug:
            print('NLayerDiscriminator intermediate output')

        input_chunked = torch.chunk(input, self.n_split, dim=1)
        output_chunked = [self.net(x) for x in input_chunked]
        output_cat = torch.cat(output_chunked, dim=1)
        output_cat = self.swap_to_last(output_cat)
        output_cat = self.dense_layer(output_cat)
        output_cat = self.swap_to_first(output_cat)

        # This code was used when we used a ModuleList instead of a Sequential object.
        # for i_layer in self.model:
        #     input = i_layer(input)
        #     if self.debug and self.debug_display_counter == 0:
        #         print(f'\t Output layer shape ', input.shape)

        self.debug_display_counter += 1
        return output_cat


class NLayerDiscriminatorMSG(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, **kwargs):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.ModuleList(sequence)

        self.debug_display_counter = 0
        self.debug = kwargs.get('debug', False)

    def forward(self, input, multiscale_input, **kwargs):
        """Standard forward."""
        if self.debug_display_counter == 0 and self.debug:
            print('NLayerDiscriminator intermediate output')

        msg_temp = multiscale_input.pop()
        msg_shape = msg_temp.shape
        for i_layer in self.model:
            input_shape = input.shape[-2:]
            if input_shape == msg_shape:
                input += msg_temp
                try:
                    msg_temp = multiscale_input.pop()
                    msg_shape = msg_temp.shape
                except IndexError:
                    msg_temp = []
                    msg_shape = None

            input = i_layer(input)
            if self.debug and self.debug_display_counter == 0:
                print(f'\t Output layer shape ', input.shape)

        self.debug_display_counter += 1
        return input


class NLayerDiscriminator3D(nn.Module):
    """Self created 3D version of the Nlayers discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, **kwargs):
        """
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = kwargs.get('kernel_size', (2, 4, 4))
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


if __name__ == "__main__":
    # Load data
    import os
    import helper.plot_fun as hplotf
    import numpy as np
    import helper.array_transf as harray
    import torch
    import data_generator.Rx2Tx as dg_rxtx
    import data_generator.UndersampledRecon as dg_recon


    dg = dg_rxtx.DataSetSurvey2B1_all(ddata='/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels',
                                      input_shape=(2, 8, 512, 256),
                                      masked=True, concatenate_complex=False)

    dg = dg_rxtx.DataSetSurvey2B1_flavio(ddata='/home/bugger/Documents/data/simulation/prostate_mri_mrl',
                                         input_shape=(16, 256, 256),
                                         masked=True)

    dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation'
    dg = dg_recon.DataGeneratorSemireal(dir_data, transform_type='complex', complex_type='polar', shuffle=False)

    a_tens, b, c = dg.__getitem__(0)

    ny, nx = a_tens[0].shape
    n_pool_layer = 2
    start_ch = 16
    concat_ch = (start_ch) * 2 ** n_pool_layer
    concat_ny = ny / 4 ** n_pool_layer
    n_features = int(concat_ch * concat_ny ** 2)

    mod_obj = Discriminator(n_pool_layers=n_pool_layer, start_ch=start_ch, debug=True, n_features=n_features, groups=8)
    output1, output2 = mod_obj(a_tens[np.newaxis])

    # # # Test Nlayer

    import model.ResNet
    mod_obj = model.ResNet.ResnetGeneratorMSG(2, 2, block_name='convblock2d')
    res_a, res_b = mod_obj(a_tens[np.newaxis])
    len(mod_obj.model)
    len(mod_obj.msg_ind_layer)
    len(res_b)

    mod_obj = NLayerDiscriminatorMSG(input_nc=2, debug=True)
    with torch.no_grad():
        output = mod_obj(a_tens[np.newaxis], res_b)
    print(output.shape)

    mod_obj = PixelDiscriminator(input_nc=2, debug=True)
    with torch.no_grad():
        output = mod_obj(a_tens[np.newaxis])
    output.shape

    mod_obj = PixelSplitDiscriminator(input_nc=2, n_split=8, debug=True)
    with torch.no_grad():
        output = mod_obj(a_tens[np.newaxis])
    output.shape
    hplotc.SlidingPlot(output)

    mod_obj = NLayerDiscriminator3D(input_nc=2, norm_layer=torch.nn.InstanceNorm3d)
    a_tens.shape
    mod_obj(a_tens[np.newaxis]).shape

    conv_layer_config= {
                            "kernel_size": 1,
                            "stride": 1,
                            "padding": 0
                          }
    activation = "leakyrelu"
    activation_config = {
                            "negative_slope": 0.2
                          }

    model_obj = DeepPixelDiscriminator(in_chan=2, start_chan=2, n_layer=3,
                                       conv_layer_config=conv_layer_config, activation=activation,
                                       activation_config=activation_config,
                                       norm_layer=hlayer.EvoNorm2D, debug=True)
    A = torch.as_tensor(np.random.rand(1, 2, 100, 100)).float()
    with torch.no_grad():
        derp = model_obj(A)
