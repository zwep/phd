# encoding: utf-8
import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple


class DenseNetFeatures(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        densenet_pretrained_features = models.densenet161(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice1.add_module('single_layer', nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, bias=False))
        for x in range(4):
            self.slice1.add_module(str(x), densenet_pretrained_features[x])
        for x in range(4, 6):
            self.slice2.add_module(str(x), densenet_pretrained_features[x])
        for x in range(6, 8):
            self.slice3.add_module(str(x), densenet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), densenet_pretrained_features[x])
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
        dense_outputs = namedtuple("DenseOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = dense_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


if __name__ == "__main__":

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import skimage.data
    import helper.plot_class as hplotc

    model_obj = DenseNetFeatures()
#    A = np.random.rand(1, 1, 64, 64)
    A = np.moveaxis(skimage.data.astronaut()[:, :, 0:1], -1, 0)[None]
    A_tensor = torch.from_numpy(A).float()
    with torch.no_grad():
        result = model_obj(A_tensor)

    hplotc.SlidingPlot(result[2])