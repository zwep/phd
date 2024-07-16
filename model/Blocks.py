# encoding: utf-8


import torch

import torch.nn as nn
import helper_torch.layers as hlayers
import functools
from helper_torch.misc import activation_selector, module_selector


class ConvBlock1D(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob=0.2, convblock_activation='identity'):
        super().__init__()
        self.activation = convblock_activation
        self.layers = nn.Sequential(
            nn.Conv1d(in_chans, in_chans, kernel_size=3, padding=1, groups=self.groups),
            nn.BatchNorm1d(in_chans),
            activation_selector(self.activation, self.debug),
            nn.Dropout(drop_prob),
            nn.Conv1d(in_chans, out_chans, kernel_size=3, padding=1, groups=self.groups),
            nn.BatchNorm1d(out_chans),
            activation_selector(self.activation, self.debug),
            nn.Dropout(drop_prob)
        )

    def forward(self, x):
        return self.layers(x)


class ConvBlockEvo2D(nn.Module):
    def __init__(self, in_chans, out_chans=None, drop_prob=0.2, groups=1, kernel_size=3, convblock_activation='relu',
                 padding=1, **kwargs):
        super().__init__()

        self.in_chans = in_chans
        if out_chans is None:
            out_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.groups = groups
        self.debug = kwargs.get('debug', False)
        self.activation = convblock_activation

        if self.debug:
            print(f'groups - {self.groups}, chans - {self.in_chans}, activation - {self.activation}')

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size, padding=padding, groups=self.groups),
            hlayers.EvoNorm2D(input=in_chans, groups=groups),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding, groups=self.groups),
            hlayers.EvoNorm2D(input=out_chans, groups=groups),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, 'f'drop_prob={self.drop_prob})'


class ConvBlock2D(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans=None, drop_prob=0.2, groups=1, block_activation='relu',
                 block_normalization='batchnorm2d', **kwargs):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        if out_chans is None:
            out_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.groups = groups
        self.debug = kwargs.get('debug', False)
        self.activation = block_activation
        self.normalization = block_normalization

        if self.debug:
            print(f'groups - {self.groups}, chans - {self.in_chans}, activation - {self.activation}')

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1, groups=self.groups),
            module_selector(self.normalization)(in_chans),
            activation_selector(self.activation, self.debug),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, groups=self.groups),
            module_selector(self.normalization)(out_chans),
            activation_selector(self.activation, self.debug),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, 'f'drop_prob={self.drop_prob})'


class GroupedConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans=None, drop_prob=0.2, n_group=4, **kwargs):
        # Expects input of size... (?, channel, y, x)
        # Where in_chans % n_group == 0
        super().__init__()
        assert in_chans % n_group == 0, f"Requested input channels {in_chans}, groupsize {n_group}"

        self.in_chans = in_chans
        if out_chans is None:
            out_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.activation = kwargs.get('convblock_activation', 'relu')
        self.debug = kwargs.get('debug', False)
        self.n_group = n_group
        self.group_size = in_chans//self.n_group

        self.module_1 = nn.ModuleList([nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1, groups=n_group),
                               nn.BatchNorm2d(in_chans),
                               nn.ReLU(),  # Could do activation selector
                               nn.Dropout2d(drop_prob),
                               hlayers.SplitStackLayer(dim_split=1, size_split=in_chans//self.n_group, dim_cat=-1),
                               nn.Linear(n_group, n_group),
                               hlayers.SplitCatLayer(dim_split=-1, size_split=1, dim_cat=1)])

        self.module_2 = nn.ModuleList([nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, groups=n_group),
                             nn.BatchNorm2d(out_chans),
                             nn.ReLU(),  # Could do activation selector
                             nn.Dropout2d(drop_prob),
                             hlayers.SplitStackLayer(dim_split=1, size_split=out_chans//self.n_group, dim_cat=-1),
                             nn.Linear(n_group, n_group),
                             hlayers.SplitCatLayer(dim_split=-1, size_split=1, dim_cat=1)])

    def forward(self, x):
        for i_layer in self.module_1:
            x = i_layer(x)
            # print(x.shape)
        x = x[:, :, :, :, 0]  #
        for i_layer in self.module_2:
            x = i_layer(x)
            # print(x.shape)
        x = x[:, :, :, :, 0]  #
        return x


class ConvBlock2Dlowrank(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans=None, kernel_size=3, drop_prob=0.2, groups=1,
                 convblock_activation='relu', **kwargs):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        if out_chans is None:
            out_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.groups = groups
        self.debug = kwargs.get('debug', False)
        self.activation = convblock_activation

        if self.debug:
            print(f'groups - {self.groups}, chans - {self.in_chans}, activation - {self.activation}')

        self.layers = nn.Sequential(
            hlayers.xCNNlowrank(in_chans=in_chans, out_chans=in_chans, kernel_size=kernel_size, padding=1, groups=self.groups),
            nn.BatchNorm2d(in_chans),
            activation_selector(self.activation, self.debug),
            nn.Dropout2d(drop_prob),
            hlayers.xCNNlowrank(in_chans, out_chans, kernel_size=kernel_size, padding=1, groups=self.groups),
            nn.BatchNorm2d(out_chans),
            activation_selector(self.activation, self.debug),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        # print('low CNN thing', input.shape)
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlocklow(in_chans={self.in_chans}, out_chans={self.out_chans}, 'f'drop_prob={self.drop_prob})'


class ConvBlock3D(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation.
    """
    def __init__(self, in_chans, out_chans, mid_chans=None, kernel_size=3, padding=0, track_batchnorm=True,
                 normalization=nn.BatchNorm3d, **kwargs):
        super().__init__()

        if mid_chans is None:
            mid_chans = in_chans

        self.in_chans = in_chans
        self.mid_chans = mid_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(nn.Conv3d(in_chans, mid_chans, kernel_size=kernel_size, padding=padding),
                                    normalization(mid_chans, track_running_stats=track_batchnorm),
                                    nn.ReLU(),
                                    nn.Conv3d(mid_chans, out_chans, kernel_size=kernel_size, padding=padding),
                                    normalization(out_chans, track_running_stats=track_batchnorm),
                                    nn.ReLU())

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, mid_chans={self.mid_chans}, out_chans={self.out_chans})'


class DenseBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans=None, mid_chans=None, drop_prob=0.2, **kwargs):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        if mid_chans is None:
            mid_chans = in_chans
        if out_chans is None:
            out_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.activation = kwargs.get('denseblock_activation', 'relu')
        self.debug = kwargs.get('debug', False)

        self.layers = nn.Sequential(
            hlayers.SwapAxes2D(to_channel_last=True),
            nn.Linear(in_chans, mid_chans),
            hlayers.SwapAxes2D(to_channel_last=False),
            nn.BatchNorm2d(mid_chans),
            activation_selector(self.activation, self.debug),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(mid_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            activation_selector(self.activation, self.debug),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'DenseBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, 'f'drop_prob={self.drop_prob})'


class ResnetBlock2Dlowrank(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans=None, kernel_size=3, drop_prob=0.2, groups=1,
                 convblock_activation='relu', **kwargs):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        if out_chans is None:
            out_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.groups = groups
        self.debug = kwargs.get('debug', False)
        self.activation = convblock_activation

        if self.debug:
            print(f'groups - {self.groups}, chans - {self.in_chans}, activation - {self.activation}')

        self.layers = nn.Sequential(
            hlayers.xCNNlowrank(in_chans=in_chans, out_chans=in_chans, kernel_size=kernel_size, padding=1, groups=self.groups),
            nn.BatchNorm2d(in_chans),
            activation_selector(self.activation, self.debug),
            nn.Dropout2d(drop_prob),
            hlayers.xCNNlowrank(in_chans, out_chans, kernel_size=kernel_size, padding=1, groups=self.groups),
            nn.BatchNorm2d(out_chans),
            activation_selector(self.activation, self.debug),
            nn.Dropout2d(drop_prob)
        )


    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        # print('low CNN thing', input.shape)
        return input + self.layers(input)

    def __repr__(self):
        return f'ConvBlocklow(in_chans={self.in_chans}, out_chans={self.out_chans}, 'f'drop_prob={self.drop_prob})'


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, in_chans, padding_type='reflect', norm_layer=torch.nn.InstanceNorm2d, drop_prob=0.2,
                 use_bias=True, **kwargs):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super().__init__()
        self.conv_block = self.build_conv_block(in_chans=in_chans,
                                                padding_type=padding_type,
                                                norm_layer=norm_layer, drop_prob=drop_prob,
                                                use_bias=use_bias)

    def build_conv_block(self, in_chans, padding_type, norm_layer, drop_prob, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(in_chans), nn.ReLU(True)]
        if drop_prob > 0:
            conv_block += [nn.Dropout(drop_prob)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(in_chans)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class EnhancedResnetBlock(nn.Module):
    """
    source : https://arxiv.org/pdf/1707.02921.pdf
    Define an Enhanced Resnet block
    """

    def __init__(self, in_chans, padding_type='reflect', use_bias=True, **kwargs):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super().__init__()
        self.conv_block = self.build_conv_block(in_chans=in_chans,
                                                padding_type=padding_type,
                                                use_bias=use_bias)

    def build_conv_block(self, in_chans, padding_type, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=p, bias=use_bias),
                       nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=p, bias=use_bias)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


# class ResidualBlock(nn.Module):
#     def __init__(self, input_channels, output_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.input_channels = input_channels
#         self.output_channels = output_channels
#         self.stride = stride
#         self.bn1 = nn.BatchNorm2d(input_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(input_channels, output_channels // 4, 1, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(output_channels // 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(output_channels // 4, output_channels // 4, 3, stride, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(output_channels // 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(output_channels // 4, output_channels, 1, 1, bias=False)
#         self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
#
#     def forward(self, x):
#         residual = x
#         out = self.bn1(x)
#         out1 = self.relu(out)
#         out = self.conv1(out1)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn3(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#         if (self.input_channels != self.output_channels) or (self.stride != 1):
#             residual = self.conv4(out1)
#         out += residual
#         return out


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, **kwargs):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.debug = kwargs.get('debug', False)
        self.outermost = outermost
        # I dont understand it fully. But it is used/needed
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            if self.debug:
                print(",input shape", x.shape)
                print("model", self.model)
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

# For backward compatibility
# 2021/02/24 - commented this. Because it was not forward compatible....
# 2021/03/01 - commented this. NWraa
ResBlock = ResnetBlock
ResidualBlock = ResnetBlock


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import numpy as np

    A = np.random.rand(1, 1, 32, 32)
    A_tens = torch.as_tensor(A).float()

    seq_1 = ResBlock(1, 4, 8)
    seq_1.layers(A_tens)
    seq_1(A_tens).shape

    seq_2 = DenseBlock(1, 4, 8)
    seq_2(A_tens).shape

    a_tens = torch.as_tensor(np.random.rand(4, 4, 32, 32))
    a_tens.repeat((1, 3, 1, 1)).shape
    b_tens = torch.as_tensor(np.random.rand(4, 12, 32, 32))
    a_tens.expand_as(b_tens).shape

    evo_block = ConvBlockEvo2D(in_chans=1, out_chans=4, groups=1)
    evo_block(A_tens)