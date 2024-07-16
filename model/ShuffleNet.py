# encoding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
import torch.nn
import helper_torch.misc as htmisc
import helper_torch.layers as htlayers

"""
Redo work on Shuffle Net
"""


class ShuffleRun3D(torch.nn.Module):
    def __init__(self):
        super().__init__()


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = torch.nn.Conv3d(1, 2, 3)

    def forward(self, x):
        return self.conv_layer(x)


class ShuffleEncoder3D(ShuffleRun3D):
    def __init__(self):
        # Input to model is something like...
        # (batch, time, y, x)...
        # Encoder should create something like
        # (batch, time,
        super().__init__()
        # Define an encoding model.. that can be something defined here..
        self.encoding_model = Encoder()
        self.sinkhorn_layer = htlayers.SinkhornLayer()  # Apply it over a certain axis..
        self.l_slice = None

    def forward(self, input):
        x = input
        split_x = torch.chunk(x, self.n_slice, dim=1)  # Along some dimension...
        encoded_x = [self.encoding_model(x) for x in split_x]
        stacked_x = torch.stack(encoded_x, dim=1)  # Stack it again...
        # Apply Sinkhorn layer..
        x = self.sinkhorn_layer(stacked_x)
        # Maybe add some Dense layer to it as well..
        x = torch.einsum("", input, x)
        return x