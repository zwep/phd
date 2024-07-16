# encoding: utf-8

import inspect
import torch

import helper.misc as hmisc
import helper_torch.activations as hactv
import helper_torch.layers as hlayer
import random
import torch.autograd
import re
import functools
from torch.nn import init


def get_coil_position_torch(input_array):
    inp_abs = torch.sqrt(input_array[::2] ** 2 + input_array[1::2] ** 2)
    # inp_abs = input_array_abs
    inp_norm = inp_abs / (torch.norm(inp_abs, dim=0) + 1e-5)
    inp_norm[torch.isnan(inp_norm)] = 0
    inp_norm = torch.abs(inp_norm)
    inp_norm_max_x, inp_max_x = torch.max(inp_norm, dim=1)
    inp_norm_max_y, inp_max_y = inp_norm_max_x.max(dim=1)

    max_position_ind = []
    for i in range(len(inp_abs)):
        pos_x = inp_max_y[i]
        pos_y = inp_max_x[i][pos_x]
        print('Max value ', inp_norm[i][pos_y, pos_x])
        max_position_ind.append([int(pos_y), int(pos_x)])

    return max_position_ind


def get_all_parameters(list_children):
    param_children = [(str(x.__class__), list(x.parameters())) for x in list_children if len(list(x.parameters()))]
    sel_layer_name, sel_layer_param = zip(*param_children)
    # Make the naming a little bit better..
    # We only want to extract the layer name.
    sel_layer_name = [re.findall(".*\.([A-Z]\w+)'>", x)[0] if re.findall(".*\.([A-Z]\w+)'>", x) else None for x in
                      sel_layer_name]
    # Can be used to check if the amount of names has changed...
    binary_check = len(sel_layer_name) == len(sel_layer_param)
    if binary_check:
        return sel_layer_name, sel_layer_param
    else:
        return -1


def get_param_layers(layer_param, layer_name, debug=False):
    param_level = []
    for i, container in enumerate(zip(layer_name, layer_param)):
        x_name, x_param = container
        if x_name is not None:
            if 'conv2d' == x_name.lower():
                for i_layer in x_param:
                    if i_layer.ndim > 2:  # Select only the 2D kernels..
                        temp_shape = (-1,) + i_layer.shape[-2:]
                        i_layer_param = i_layer.reshape(temp_shape)
                        res = i_layer_param.norm(p=2, dim=(-2, -1))
                        param_level.append((x_name.lower() + f'_{i}', res))

            if 'linear' == x_name.lower():
                for i_layer in x_param:
                    if i_layer.ndim > 1:  # Select only the 2D kernels..
                        temp_shape = (-1,) + i_layer.shape[-2:]
                        i_layer_param = i_layer.reshape(temp_shape)
                        res = i_layer_param.norm(p=2, dim=(-2, -1))
                        param_level.append((x_name.lower() + f'_{i}', res))

    return param_level


def get_grad_layers(layer_param, layer_name, debug=False, debug_display_counter=0):
    grad_level = []
    for i, container in enumerate(zip(layer_name, layer_param)):
        x_name, x_param = container
        if x_name is not None:
            if 'conv2d' == x_name.lower():
                for i_layer in x_param:
                    if i_layer.ndim > 2:  # Select only the 2D kernels..
                        temp_shape = (-1,) + i_layer.shape[-2:]
                        if i_layer.grad is not None:
                            i_layer_grad = i_layer.grad.reshape(temp_shape)
                            res = i_layer_grad.norm(p=2, dim=(-2, -1))
                            grad_level.append((x_name.lower() + f'_{i}', res))
                            if debug and debug_display_counter < 2:
                                print(f'Norm for layer {i}', res)
                        else:
                            print('There is no gradient in layer ', x_name, i)

            if 'linear' == x_name.lower():
                for i_layer in x_param:
                    if i_layer.ndim > 1:  # Select only the 2D kernels..
                        temp_shape = (-1,) + i_layer.shape[-2:]
                        if i_layer.grad is not None:
                            i_layer_grad = i_layer.grad.reshape(temp_shape)
                            res = i_layer_grad.norm(p=2, dim=(-2, -1))
                            grad_level.append((x_name.lower() + f'_{i}', res))
                            if debug and debug_display_counter < 2:
                                print(f'Norm for layer {i}', res)
                        else:
                            print('There is no gradient in layer ', x_name, i)


    return grad_level


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)

    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def extract_patches(features, size, stride, normalize, epsilon=10e-10):
    """
    Arguments:
        features: a float tensor with shape [C, H, W].
        size: an integer, size of the patch.
        stride: an integer.
        normalize: a boolean.
    Returns:
        a float tensor with shape [M, C, size, size],
        where M = n * m, n = 1 + floor((H - size)/stride),
        and m = 1 + floor((W - size)/stride).
    """
    C = features.size(0)
    patches = features.unfold(dimension=1, size=size, step=stride).unfold(dimension=2, size=size, step=stride)
    # it has shape [C, n, m, size, size]

    # get the number of patches
    n, m = patches.size()[1:3]
    M = n * m

    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(M, C, size, size)
    if normalize:
        norms = patches.view(M, -1).norm(p=2, dim=1)  # shape [M]
        patches /= (norms.view(M, 1, 1, 1) + epsilon)

    return patches


def activation_selector(activation_name, debug=False, config=None):
    # This function gives back the instance of the object..
    # All other functions dont do this...
    # Maybe, someday, I will also implement that here. That day is not today
    # Used to easily plant some activation function
    if debug:
        print('Getting activation, ', activation_name)

    if config is None:
        config = {}

    activation_name = activation_name.lower()
    hactv_dict = {k.lower(): v for k, v in inspect.getmembers(hactv, inspect.isclass)}
    nn_dict = {k.lower(): v for k, v in inspect.getmembers(torch.nn.modules.activation, inspect.isclass)}

    if activation_name in hactv_dict.keys():
        output_actv_layer = hactv_dict[activation_name](**config)
    elif activation_name in nn_dict.keys():
        output_actv_layer = nn_dict[activation_name](**config)
    else:
        output_actv_layer = None
        print('Unknown activation name ', activation_name)
        hmisc.print_dict(hactv_dict)
        hmisc.print_dict(nn_dict)

    return output_actv_layer


def block_selector(block_name, debug=False):
    # Used to easily plant some activation function
    # Need to do this to avoid circular reference
    from model import Blocks
    if debug:
        print('Getting block, ', block_name)

    block_name = block_name.lower()
    hblock_dict = {k.lower(): v for k, v in inspect.getmembers(Blocks, inspect.isclass)}

    if block_name in hblock_dict.keys():
        output_block = hblock_dict[block_name]
    else:
        output_block = None
        print('Unknown block name ', block_name)
        hmisc.print_dict(hblock_dict)

    return output_block


def module_selector(module_name, debug=False):
    # Used to easily plant some activation function
    if debug:
        print('Getting module, ', module_name)

    module_name = module_name.lower()
    hlayer_dict = {k.lower(): v for k, v in inspect.getmembers(hlayer, inspect.isclass)}
    nn_dict = {k.lower(): v for k, v in inspect.getmembers(torch.nn, inspect.isclass)}

    if module_name in hlayer_dict.keys():
        output_module = hlayer_dict[module_name]
    elif module_name in nn_dict.keys():
        output_module = nn_dict[module_name]
    else:
        output_module = None
        print('Unknown module name ', module_name)
        hmisc.print_dict(hlayer_dict)
        hmisc.print_dict(nn_dict)

    return output_module


def normalization_selector(norm_name, debug=False):
    if debug:
        print('Getting block, ', norm_name)

    pool_name = norm_name.lower()
    hpool_dict = {k.lower(): v for k, v in inspect.getmembers(torch.nn, inspect.isclass)}

    if pool_name in hpool_dict.keys():
        output_block = hpool_dict[pool_name]
    else:
        output_block = None
        print('Unknown block name ', pool_name)
        hmisc.print_dict(hpool_dict)

    return output_block


class ReplayBuffer:
    # I have no idea what this is yet.. But some soft of buffer.
    # https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/utils.py
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = {}

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            xsize = element.shape[-1]
            if len(self.data) < self.max_size:
                self.data.setdefault(xsize, [])
                self.data[xsize].append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    n_max = len(self.data[xsize]) - 1
                    i = random.randint(0, n_max)
                    to_return.append(self.data[xsize][i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.autograd.Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        """

        :param n_epochs:
        :param offset:
        :param decay_start_epoch:
        """
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        """

        :param epoch:
        :return:
        """
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def get_all_children(model, children=None):
    if children is None:
        children = []

    temp_children = list(model.children())

    if len(temp_children):
        for i in temp_children:
            get_all_children(i, children)
    else:
        children.append(model)
    return children


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    This one is used in the Pix2pix models.

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(torch.nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return hactv.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f'\t weight initialization: {init_type}')
    net.apply(init_func)  # apply the initialization function <init_func>

