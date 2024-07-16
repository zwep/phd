# encoding: utf-8

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, set_weights=True):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        # OMG I forgot to set the weights of the conv layer....
        # Lijkt (visuele inspectie) niet SUPER veel uit te maken...
        conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, bias=False)
        if set_weights:
            conv_weights = torch.from_numpy(np.ones((3, 1, 1, 1))).float()
            conv_layer.weight = torch.nn.Parameter(conv_weights, requires_grad=False)
        self.slice1.add_module('single_layer', conv_layer)
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


if __name__ == "__main__":
    model_obj = Vgg16()
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import skimage.data
    import helper.plot_class as hplotc
    A = np.random.rand(1, 1, 64, 64)
    A = np.moveaxis(skimage.data.astronaut()[:, :, 0:1], -1, 0)[None]
    A_tensor = torch.from_numpy(A).float()
    result = model_obj(A_tensor)
    hplotc.ListPlot([[x[0].sum(axis=0) for x in result]])