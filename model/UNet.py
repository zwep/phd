# encoding: utf-8

"""
source: https://github.com/facebookresearch/fastMRI/blob/master/models/unet/unet_model.py

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
from model.Blocks import ConvBlock2D
import helper_torch.misc as htmisc


class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, groups=1, **kwargs):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        block_name = kwargs.get('block_name', 'ConvBlock2D')
        block = htmisc.block_selector(block_name)

        final_activation = kwargs.get('final_activation', 'identity')
        print('first', chans, in_chans)
        self.down_sample_layers = nn.ModuleList([block(in_chans=in_chans, out_chans=chans,
                                                       drop_prob=drop_prob, groups=groups)])
        ch = chans
        for i in range(num_pool_layers - 1):
            print(ch)
            self.down_sample_layers += [block(in_chans=ch, out_chans=ch * 2, drop_prob=drop_prob)]
            ch *= 2
        self.conv = block(in_chans=ch, out_chans=ch, drop_prob=drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            print(ch)
            self.up_sample_layers += [block(in_chans=ch * 2, out_chans=ch // 2, drop_prob=drop_prob)]
            ch //= 2

        self.up_sample_layers += [block(in_chans=ch * 2, out_chans=ch, drop_prob=drop_prob)]
        self.conv2 = nn.Sequential(block(in_chans=ch, out_chans=ch // 2, kernel_size=1),
                                   block(in_chans=ch // 2, out_chans=out_chans, kernel_size=1),
                                   block(in_chans=out_chans, out_chans=out_chans, kernel_size=1))
        self.final_actv = htmisc.activation_selector(final_activation)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input

        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for i_count, layer in enumerate(self.up_sample_layers):
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)

        output = self.conv2(output)
        return self.final_actv(output)


class UnetModelShadow(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, groups=1, **kwargs):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.debug = kwargs.get('debug')
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        block_name = kwargs.get('block_name', 'ConvBlock2D')
        block = htmisc.block_selector(block_name)
        self.down_sample_layers = nn.ModuleList([block(in_chans=in_chans, out_chans=chans,
                                                       drop_prob=drop_prob, groups=2)])
        self.down_sample_layers += nn.ModuleList([block(in_chans=chans, out_chans=chans,
                                                       drop_prob=drop_prob, groups=2)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers +=[block(in_chans=ch, out_chans=ch * 2, drop_prob=drop_prob)]
            ch *= 2
        self.conv = block(in_chans=ch, out_chans=ch, drop_prob=drop_prob)

        self.up_sample_layers = nn.ModuleList()
        self.conv_layer_list = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [block(in_chans=ch * 2, out_chans=ch // 2, drop_prob=drop_prob)]
            self.conv_layer_list += [torch.nn.Conv2d(ch, ch, kernel_size=1)]  # THIS IS EXTRA
            ch //= 2

        self.up_sample_layers += [block(in_chans=ch * 2, out_chans=ch, drop_prob=drop_prob)]
        self.conv_layer_list += [torch.nn.Conv2d(ch, ch, kernel_size=1)]  # THIS IS EXTRA
        self.additional_convblock = nn.Sequential(block(in_chans=ch, out_chans=ch, drop_prob=drop_prob, groups=2))
        self.final_convblock = nn.Sequential(block(in_chans=ch, out_chans=out_chans, drop_prob=drop_prob, groups=1))

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input

        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            if self.debug:
                print(layer, output.shape)
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for i_count, layer in enumerate(self.up_sample_layers):
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = self.conv_layer_list[i_count](output)  # THIS IS EXTRA
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)

        output = self.additional_convblock(output)
        output = self.final_convblock(output)
        return output


""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    # https://github.com/milesial/Pytorch-UNet
    def __init__(self, n_channels, n_classes, bilinear=True, **kwargs):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    import importlib
    import model.UNet as unet_model
    import torchsummary
    importlib.reload(unet_model)
    in_chans = 1
    out_chans = 1
    chans = 4
    num_pool_layers = 3
    drop_prob = 0.2

    import data_generator.Rx2Tx as data_gen_rx2tx
    dir_data = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels'

    dg_gen_rx2tx = data_gen_rx2tx.DataSetSurvey2B1_single(input_shape=(1, 512, 256),
                                                     ddata=dir_data, input_is_output=False,
                                                     transform_type='angle', number_of_examples=1, debug=True,
                                                     complex_type='polar')
    a, b = dg_gen_rx2tx.__getitem__(0)
    mod_obj = UnetModel(in_chans=in_chans, out_chans=out_chans, chans=chans, debug=True,
                                   num_pool_layers=2, drop_prob=drop_prob, block_name='convblock2dlow')
    mod_obj.forward(a[None, :,])
    a.shape

    from torch.utils.data import DataLoader
    dl_rx2tx = DataLoader(dg_gen_rx2tx)
    for a, b in dl_rx2tx:
        mod_obj.forward(a)


    importlib.reload(unet_model)
    in_chans = 2
    out_chans = 2
    chans = 4
    num_pool_layers = 2
    drop_prob = 0.2
    A = unet_model.UnetModel(in_chans=in_chans, out_chans=out_chans, chans=chans,
                                   num_pool_layers=num_pool_layers, drop_prob=drop_prob)

    A.forward(a[None, ...])
