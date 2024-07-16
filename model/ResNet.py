# encoding: utf-8

import functools
import torch
import torch.nn as nn
import model.Blocks as Blocks
import numpy as np
import helper_torch.layers as hlayers
import helper_torch.misc as htmisc

# from torchvision.models import resnet50, ResNet50_Weights
# resnet_obj = resnet50(weights="IMAGENET1K_V2")
# resnet_obj.eval()
# # dir(resnet_obj.features
# weights = ResNet50_Weights.DEFAULT
# preprocess = weights.transforms()
# # Apply it to the input image

#
# import skimage.data
# import torch
# import numpy as np
# A = skimage.data.astronaut()[:, :, 0]
# A = np.array([A, A, A])
# A_tens = torch.from_numpy(A)
# img_transformed = preprocess(A_tens)
# import helper.plot_class as hplotc
# hplotc.ListPlot([A, A_tens])

class ResnetFeature(nn.Module):
    def __int__(self, requires_grad=False):
        pass

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, drop_prob=0.4, n_blocks=6, padding_type='reflect', **kwargs):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super().__init__()
        use_dropout = False
        if drop_prob > 0:
            use_dropout = True

        final_activation_name = kwargs.get('final_activation', 'tanh')
        conv_layer_name = kwargs.get('conv_layer', 'conv2d')
        block_name = kwargs.get('block_name', 'resnetblock')
        n_downsampling = kwargs.get('downsampling', 2)
        block = htmisc.block_selector(block_name)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 htmisc.module_selector(conv_layer_name)(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [htmisc.module_selector(conv_layer_name)(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                block(in_chans=ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      # Added a 1x1 kernel to smoothen out the blocks from the Conv Transpose..(that is the idea)
                      # nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [htmisc.module_selector(conv_layer_name)(ngf, output_nc, kernel_size=7, padding=0)]
        final_activation = htmisc.activation_selector(final_activation_name)
        model += [final_activation]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetGeneratorMSG(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, drop_prob=0.4, n_layer=2, n_blocks=6, padding_type='reflect', **kwargs):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super().__init__()
        use_dropout = False
        if drop_prob > 0:
            use_dropout = True

        final_activation_name = kwargs.get('final_activation', 'tanh')
        conv_layer_name = kwargs.get('conv_layer', 'conv2d')
        block_name = kwargs.get('block_name', 'resnetblock')
        block = htmisc.block_selector(block_name)
        # Used to keep track when to take the intermediate layer output..
        self.msg_ind_layer = []
        # Used to apply to the intermediate layer output (1x1 convs)
        self.conv_layer_11 = []

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 htmisc.module_selector(conv_layer_name)(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        self.msg_ind_layer = [False, False, False, False]

        n_downsampling = n_layer
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [htmisc.module_selector(conv_layer_name)(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            self.msg_ind_layer += [False, False, False]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                block(in_chans=ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
            self.msg_ind_layer += [False]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            n_chan = int(ngf * mult / 2)
            model += [nn.ConvTranspose2d(ngf * mult, n_chan,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(n_chan),
                      nn.ReLU(True)]
            self.msg_ind_layer += [False, False, True]
            self.conv_layer_11 += [nn.Conv2d(in_channels=n_chan, out_channels=1, kernel_size=1)]

        model += [nn.ReflectionPad2d(3)]
        model += [htmisc.module_selector(conv_layer_name)(ngf, output_nc, kernel_size=7, padding=0)]
        final_activation = htmisc.activation_selector(final_activation_name)
        model += [final_activation]

        # This way it should be properly adressed to a GPU..
        self.conv_layer_11 = nn.ModuleList(self.conv_layer_11)
        self.msg_ind_layer += [False, False, False]
        self.model = nn.ModuleList(model)

    def forward(self, input):
        """Standard forward"""
        msg_layer = []
        msg_counter = 0
        for i, i_layer in enumerate(self.model):
            input = i_layer(input)
            if self.msg_ind_layer[i]:
                # Apply a 1x1 conv before adding it to the list
                print('msg counter ', msg_counter)
                msg_layer.append(self.conv_layer_11[msg_counter](input))
                msg_counter += 1
        return input, msg_layer


class ResnetSplit(nn.Module):
    """
    TODO need to work on this one...
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, drop_prob=0.4, n_blocks=6, padding_type='reflect', **kwargs):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super().__init__()
        use_dropout = False
        if drop_prob > 0:
            use_dropout = True

        final_activation_name = kwargs.get('final_activation', 'tanh')
        conv_layer_name = kwargs.get('conv_layer', 'conv2d')
        block_name = kwargs.get('block_name', 'resnetblock')
        block = htmisc.block_selector(block_name)
        n_layer = kwargs.get('n_layer', 2)
        self.n_split = kwargs.get('n_split', 1)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.swap_ax_to_last = hlayers.SwapAxes2D(to_channel_last=True)
        self.swap_ax_to_first = hlayers.SwapAxes2D(to_channel_last=False)
        self.pre_model = torch.nn.ModuleList([hlayers.LearnableRotation() for _ in range(self.n_split)])

        model = [nn.ReflectionPad2d(3),
                 htmisc.module_selector(conv_layer_name)(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_layer):  # add downsampling layers
            mult = 2 ** i
            model += [htmisc.module_selector(conv_layer_name)(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_layer
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                block(in_chans=ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        for i in range(n_layer):  # add upsampling layers
            mult = 2 ** (n_layer - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [htmisc.module_selector(conv_layer_name)(ngf, output_nc, kernel_size=7, padding=0)]
        final_activation = htmisc.activation_selector(final_activation_name)
        model += [final_activation]

        self.model = nn.Sequential(*model)

        self.debug = kwargs.get('debug', False)
        self.debug_display_counter = 0
        self.debug_model_name = 'ResNet Split'

    def forward(self, input):
        """Standard forward"""

        # This is not a block per se....
        input_tensor_list = torch.chunk(input, self.n_split, dim=1)

        if self.debug and self.debug_display_counter == 0:
            print(f'{self.debug_model_name} - after split')
            print('length input...', len(input_tensor_list))
            print('shape input...', input_tensor_list[0].shape)

        # Here we have a setup for Learnable ROtation
        temp_premodel = []
        for i, x in enumerate(input_tensor_list):
            x_temp = self.swap_ax_to_last(x)
            if self.debug and self.debug_display_counter == 0:
                print(f'{self.debug_model_name} - swap ax to last')
                print('length input...', len(x_temp))
                print('shape input...', x_temp.shape)

            x_temp = self.pre_model[i](x_temp)
            x_temp = self.swap_ax_to_first(x_temp)
            temp_premodel.append(x_temp)

        input_tensor_list = [self.model(x) for i, x in enumerate(input_tensor_list)]
        res = torch.cat(input_tensor_list, dim=1)

        self.debug_display_counter += 1

        return res

#
# class ResNet50(torchvision_models.ResNet):
#     def __init__(self, block=None, layers=None,
#                  num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None):
#         # Other way to define default values...
#         if block is None:
#             block = torchvision_models.resnet.Bottleneck
#         if layers is None:
#             layers = [3, 4, 6, 3]
#
#         super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual,
#                  groups=groups, width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation,
#                  norm_layer=norm_layer)
#
#     # OVerwrite this stuff to avoid the FC layers
#     def _forward_impl(self, x):
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         print('x.shape ', x.shape)
#         x = self.layer1(x)
#         print('x.shape ', x.shape)
#         x = self.layer2(x)
#         print('x.shape ', x.shape)
#         x = self.layer3(x)
#         print('x.shape ', x.shape)
#         x = self.layer4(x)
#         print('x.shape ', x.shape)
#         x = self.avgpool(x)
#         return x


# model = ResNet50(replace_stride_with_dilation=[True, True, True])
# A_inp = torch.from_numpy(np.random.rand(1, 3, 256, 256)).float()
# res = model(A_inp)
# res.shape
if __name__ == "__main__":
    import torch
    import numpy as np
    import data_generator.UndersampledRecon as data_gen
    import importlib
    import helper.plot_fun as hplotf

    importlib.reload(data_gen)
    dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation_rxtx'
    A = data_gen.DataGeneratorSemireal(dir_data, transform_type='complex', complex_type='polar', shuffle=False,
                                       dataset_type='test')
    a_coarse, b_coarse = A.__getitem__(0)
    hplotf.plot_3d_list(b_coarse)

    config_resnet = {"input_nc": 1, "output_nc": 1, "n_blocks": 50, "downsampling": 4, "ngf": 16, "drop_prob": 0.1, "normalization_layer": "EvoNorm2D", "conv_layer": "Conv2d", "final_activation": "identity", "block_name": "resblock", "padding_type": "reflect"}
    model_obj = ResnetGenerator(**config_resnet)
    model_parameters = filter(lambda p: p.requires_grad, model_obj.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of parameters', params, end='\n\n')


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    count_parameters(model_obj)

    mod_obj = ResnetGenerator(2, 2, downsampling=4)
    A = np.random.rand(1, 2, 500, 500)
    A_tens = torch.from_numpy(A).float()
    print('Before' , A_tens.shape)
    with torch.no_grad():
        res = mod_obj(A_tens)
    print(res.shape)

    print(res.shape)
    print(mod_obj)
    import helper.plot_fun as hplotf
    hplotf.plot_3d_list(res)
    # hplotf.plot_3d_list(A-res)
    # Check Resnet parameter clipping...

    mod_obj = ResnetGeneratorMSG(2, 2, block_name='convblock2d')
    res_a, res_b = mod_obj(a_coarse[np.newaxis])
    len(mod_obj.model)
    len(mod_obj.msg_ind_layer)
    len(res_b)

    mod_obj = ResnetSplit(2, 2, ngf=16, debug=True, n_split=8)
    res = mod_obj(a_coarse[np.newaxis])
    hplotf.plot_3d_list(res.detach().numpy())
# input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, drop_prob=0.4, n_blocks=6, padding_type='reflect', **kwargs):
    temp = ResnetGenerator(input_nc=1, output_nc=1, ngf=16, n_blocks=18, downsampling=4,
                           block_name='ResnetBlock2Dlowrank')

    model_parameters = filter(lambda p: p.requires_grad, temp.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)