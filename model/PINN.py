"""
Here is an example of a model that I copied  from zeh interwebs

got it from this github repo

https://github.com/okada39/pinn_burgers/tree/e9f63108c6b469cd3d21dae6840d1ae699b84b1b

Tried to find a differnet one.. but yeah..
Ah well
"""

import helper_torch.misc as hmisc
import torch.nn as nn
import torch
import numpy as np


class Network(nn.Module):
    """
    Build a physics informed neural network (PINN) model for Burgers' equation.
    """
    def __init__(self,  num_inputs=2, layers=(32, 16, 32), activation='tanh', num_outputs=1):
        super().__init__()
        self.num_inputs = num_inputs
        self.layers = layers
        self.activation = activation
        self.num_outputs = num_outputs
        self.model_obj = self.get_model()

    def get_model(self):
        in_features = self.num_inputs
        out_features = self.layers[0]
        layer_list = []
        for layer in self.layers[1:]:
            temp_dense = nn.Linear(in_features=in_features, out_features=out_features)
            temp_activation = hmisc.activation_selector(self.activation)
            layer_list.append(temp_dense)
            layer_list.append(temp_activation)

            in_features = out_features
            out_features = layer
        # UGLY
        temp_dense = nn.Linear(in_features=in_features, out_features=out_features)
        temp_activation = hmisc.activation_selector(self.activation)
        layer_list.append(temp_dense)
        layer_list.append(temp_activation)

        # output layer
        outputs = nn.Linear(in_features=self.layers[-1], out_features=self.num_outputs)
        layer_list.append(outputs)

        return nn.ModuleList(layer_list)

    def forward(self, x):
        temp = x
        for i_layer in self.model_obj:
#            print('layer', i_layer)
 #           print('shape ', temp.shape)
            temp = i_layer(temp)
        return temp


if __name__ == "__main__":
    A = torch.from_numpy(np.random.rand(1, 1, 2)).float()
    model_obj = Network()




