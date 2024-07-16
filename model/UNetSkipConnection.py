# encoding: utf-8

import numpy as np
import os
import importlib

import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc

import torch.nn as nn
import model.Blocks as Blocks


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, **kwargs):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()
        debug = kwargs.get('debug', False)
        # construct unet structure
        unet_block = Blocks.UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, debug=debug)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = Blocks.UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, debug=debug)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = Blocks.UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, debug=debug)
        unet_block = Blocks.UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, debug=debug)
        unet_block = Blocks.UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, debug=debug)
        self.model = Blocks.UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, debug=debug)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


if __name__ == "__main__":
    res = UnetGenerator(2, 2, 3)

