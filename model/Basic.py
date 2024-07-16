# encoding: utf-8

import data_generator.Rx2Tx as data_gen
import helper_torch.activations as pactivation
import torch # import main library
import torch.nn
import torch.nn.modules.loss as ploss
import importlib
importlib.reload(pactivation)
import torch.nn.functional as F # import torch functions

"""

"""
import helper_torch.misc as htmisc
import model.Blocks as Blocks
import helper_torch.layers as hlayer


class DenseModel(torch.nn.Module):
    def __init__(self, ndim_start, ndim_hidden, ndim_out=2, actv='identity', **kwargs):
        super().__init__()
        if isinstance(ndim_hidden, int):
            n_layer = kwargs.get('n_layer', 5)
            ndim_hidden = [ndim_hidden] * n_layer

        derp = []
        n_layer = len(ndim_hidden)
        temp = torch.nn.Sequential(torch.nn.Linear(in_features=ndim_start, out_features=ndim_hidden[0]),
                                   htmisc.activation_selector(actv))
        derp.append(temp)
        for i in range(0, n_layer-1):
            temp = torch.nn.Sequential(torch.nn.Linear(in_features=ndim_hidden[i], out_features=ndim_hidden[i+1]),
                                       htmisc.activation_selector(actv))
            derp.append(temp)

        temp = torch.nn.Sequential(torch.nn.Linear(in_features=ndim_hidden[n_layer-1], out_features=ndim_out),
                                   htmisc.activation_selector('identity'))
        derp.append(temp)
        self.layers = torch.nn.Sequential(*derp)

    def forward(self, x):
        return self.layers(x)


class SimpleModelConv1D(torch.nn.Module):
    def __init__(self, start_chan, out_chan, n_layer=5, actv='identity'):
        super().__init__()
        derp = []
        temp = Blocks.ConvBlock1D(in_chans=start_chan, out_chans=2, convblock_activation='identity')
        derp.append(temp)
        for i in range(1, n_layer-1):
            temp = Blocks.ConvBlock1D(in_chans=2**i, out_chans=2**(i+1), convblock_activation=actv)
            derp.append(temp)

        temp = Blocks.ConvBlock1D(in_chans=2**(i+1), out_chans=out_chan, convblock_activation='identity')
        derp.append(temp)
        self.layers = torch.nn.Sequential(*derp)

    def forward(self, x):
        return self.layers(x)


class SimpleModelConv2D(torch.nn.Module):
    def __init__(self, in_chan, start_chan, out_chan, n_layer=5, actv='identity', group_list=1):
        super().__init__()
        if isinstance(group_list, int):
            group_list = [group_list] * n_layer

        derp = []
        temp = Blocks.ConvBlock2D(in_chans=in_chan, out_chans=start_chan, convblock_activation='identity', groups=group_list[0])
        print('group size', group_list[0])
        derp.append(temp)
        for i in range(1, n_layer):
            temp = Blocks.ConvBlock2D(in_chans=start_chan * 2 ** (i-1), out_chans=start_chan * 2 ** (i), convblock_activation=actv, groups=group_list[i])
            print('group size', group_list[i])
            derp.append(temp)

        temp = Blocks.ConvBlock2D(in_chans=start_chan * 2 ** (n_layer-1), out_chans=out_chan, convblock_activation='identity', groups=group_list[n_layer-1])
        print('group size', group_list[n_layer-1])
        derp.append(temp)
        self.layers = torch.nn.Sequential(*derp)

    def forward(self, x):
        return self.layers(x)


class DenseModelV2(torch.nn.Module):
    def __init__(self, ndim_start, ndim_hidden, ndim_out=2, actv='identity', **kwargs):
        super().__init__()
        if isinstance(ndim_hidden, int):
            n_layer = kwargs.get('n_layer', 5)
            ndim_hidden = [ndim_hidden] * n_layer

        derp = []
        n_layer = len(ndim_hidden)
        temp = torch.nn.Sequential(torch.nn.Linear(in_features=ndim_start, out_features=ndim_hidden[0]),
                                   htmisc.activation_selector(actv),
                                   hlayer.Transpose2D(),
                                   torch.nn.Linear(in_features=ndim_start, out_features=ndim_hidden[0]),
                                   htmisc.activation_selector(actv),
                                   hlayer.Transpose2D())
        derp.append(temp)
        for i in range(0, n_layer-1):
            temp = torch.nn.Sequential(torch.nn.Linear(in_features=ndim_hidden[i], out_features=ndim_hidden[i+1]),
                                       htmisc.activation_selector(actv),
                                       hlayer.Transpose2D(),
                                       torch.nn.Linear(in_features=ndim_hidden[i], out_features=ndim_hidden[i+1]),
                                       htmisc.activation_selector(actv),
                                       hlayer.Transpose2D())
            derp.append(temp)

        temp = torch.nn.Sequential(torch.nn.Linear(in_features=ndim_hidden[n_layer-1], out_features=ndim_out),
                                   htmisc.activation_selector('identity'),
                                   hlayer.Transpose2D(),
                                   torch.nn.Linear(in_features=ndim_hidden[n_layer-1], out_features=ndim_out),
                                   htmisc.activation_selector('identity'),
                                   hlayer.Transpose2D())
        derp.append(temp)
        self.layers = torch.nn.Sequential(*derp)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    import torchviz
    import torch
    import numpy as np
    A = DenseModel(ndim_start=50, ndim_hidden=[25, 10, 5, 10, 25], ndim_out=50, actv='identity')
    B = torch.as_tensor(np.random.rand(1, 1, 50, 50)).float()
    Z = A(B)
    torchviz.make_dot(Z, params=dict(A.parameters())).render("attached", format="png")
    print(A.layer_list)

    import torch.nn as nn
    model = nn.Sequential()
    model.add_module('W0', nn.Linear(8, 16))
    model.add_module('tanh', nn.Tanh())
    model.add_module('W1', nn.Linear(16, 1))

    x = torch.randn(1, 8)

    torchviz.make_dot(model(x), params=dict(model.named_parameters()))



