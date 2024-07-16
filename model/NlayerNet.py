
import numpy as np
import torch
import torch.nn as nn
import helper_torch.misc as htmisc


class NLayerNet(nn.Module):
    def __init__(self, n_start, n_layer, n_hidden, activation, n_end=None, scale_factor=1.2):
        super().__init__()
        self.n_start = n_start
        self.n_end = n_end
        if n_end is None:
            self.n_end = n_start
        self.n_layer = n_layer
        self.n_hidden = n_hidden
        self.activation = htmisc.activation_selector(activation)
        # Factor by which the size of the hidden layers is increased
        self.scale_factor = scale_factor
        self.model_layers = self.get_model()

    def get_model(self):
        up_layer = self.n_layer // 2
        same_layer = 0
        if self.n_layer % 2:
            # Now we are odd and need to add an extra layer
            same_layer = 1

        layer_1 = nn.Linear(self.n_start, self.n_hidden)
        up_list = [layer_1]
        temp_hidden = self.n_hidden
        for i_up in range(up_layer):
            new_hidden = int(np.ceil(temp_hidden * self.scale_factor))
            temp = nn.Linear(temp_hidden, new_hidden)
            temp_hidden = new_hidden
            up_list.append(temp)
            up_list.append(self.activation)

        same_list = []
        for i_same in range(same_layer):
            temp = nn.Linear(temp_hidden, temp_hidden)
            same_list.append(temp)
            same_list.append(self.activation)

        down_list = []
        for i_up in range(up_layer):
            new_hidden = int(np.floor(temp_hidden * (1/self.scale_factor)))
            temp = nn.Linear(temp_hidden, new_hidden)
            temp_hidden = new_hidden
            down_list.append(temp)
            down_list.append(self.activation)

        temp = nn.Linear(temp_hidden, self.n_end)
        down_list.append(temp)

        torch_module_list = nn.ModuleList(up_list + same_list + down_list)
        return torch_module_list

    def forward(self, x):
        temp_x = x
        for i_layer in self.model_layers:
            temp_x = i_layer(temp_x)
        return temp_x


if __name__ == "__main__":
    model_obj = NLayerNet(n_start=1, n_layer=2, n_hidden=300, n_end=1, activation='relu')
    a_tens = torch.from_numpy(np.ones((1, 1))).float()
    res = model_obj(a_tens)
