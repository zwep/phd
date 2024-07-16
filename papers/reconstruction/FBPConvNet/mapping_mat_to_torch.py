import helper.plot_class as hplotc
import torchsummary
import os
import helper.array_transf as harray
import torch
from model.FBPConvnet import FBPCONVNet
import numpy as np
import objective_helper.reconstruction as obj_helper
from objective_helper.reconstruction import DWEIGHTS_FBPCONVNET
import helper.misc as hmisc
import scipy.io

"""
The lines that are printed here helped us to create a mapping from the matlab file to torch

This mapping is stored to



"""

# Create model
model_obj = FBPCONVNet()


# Check what the original key-value pairs are...
random_state_dict = model_obj.state_dict()
for ikey, ivalue in random_state_dict.items():
    print(ikey, '\t', ', '.join([str(x) for x in ivalue.shape]))

# Load weights..
mat_obj = scipy.io.loadmat(DWEIGHTS_FBPCONVNET)
layer_stuff = mat_obj['net']['layers'][0][0]
n_layers = layer_stuff.shape[-1]
for ii in range(n_layers):
    ii_layer = layer_stuff[0][ii]
    layer_name = ii_layer['name'][0][0][0]
    if len(ii_layer['weights'][0][0]):
        n_channel = ii_layer['weights'][0][0].shape[-1]
        for i_channel in range(n_channel):
            weight_shape = ii_layer['weights'][0][0][0][i_channel].shape
            print(ii, '\t', i_channel, '\t', layer_name, '\t', weight_shape)
            if len(weight_shape) == 2 and weight_shape[-1] > 1:
                print('None \t None \t None \t None')
                print('None \t None \t None \t None')
