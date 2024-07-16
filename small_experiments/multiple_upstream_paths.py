

"""
Here we are going to check whether I can get my model strategy to work
"""


import torch
import torch.nn as nn
import numpy as np
import helper.plot_class as hplotc
import helper.plot_fun as hplotf

x = np.arange(-3, 3, 0.1)
y = np.arange(-3, 3, 0.1)
X, Y = np.meshgrid(x, y)
mu_x = mu_y = 0
std_x = std_y = 1
Z = np.exp(-((X - mu_x) ** 2/(2 * std_x ** 2) + (Y - mu_y) ** 2/(2 * std_y ** 2)))
W = np.arcsinh(1-Z)
# Visualize the new targets..
hplotc.SlidingPlot(Z, ax_3d=True)
hplotc.SlidingPlot(W, ax_3d=True)

hplotc.close_all()
# Now create a model... the input data will come later

import helper_torch.misc as htmisc
import helper.misc as hmisc
import matplotlib.pyplot as plt

def get_grad_fig(x):
    x_switch = hmisc.change_list_order(x)
    n_layers = len(x_switch)

    fig, ax = plt.subplots(1, n_layers, figsize=(20, 15))
    ax = ax.ravel()
    temp_max_y = np.max(x_switch)

    if np.isfinite(temp_max_y):
        temp_max_y += 0.05 * temp_max_y
    else:
        temp_max_y = 1

    for i in range(n_layers):
        temp_min, temp_mean, temp_max = zip(*x_switch[i])
        ax[i].plot(temp_min, 'b', alpha=0.5)
        ax[i].plot(temp_mean, '-.k')
        ax[i].plot(temp_max, 'r', alpha=0.5)
        ax[i].set_ylim(0, temp_max_y)

class DummyModel(nn.Module):
    def __init__(self, n_split=2):
        super().__init__()

        model = [htmisc.block_selector('convblock2d')(1, 16), htmisc.block_selector('convblock2d')(16, 1)]
        self.n_split = n_split
        self.seq_model = torch.nn.ModuleList([nn.Sequential(*model) for _ in range(n_split)])

    def forward(self, x):
        x_split = torch.chunk(x, self.n_split, dim=1)
        res_split = [self.seq_model[i](x_split[i]) for i in range(self.n_split)]
        res = torch.cat(res_split, dim=1)
        return res


import copy

class DummyModel2(nn.Module):
    """
    Here we have found that we indeed need multiple model definitions carried out.
    Otherwise we end up with the same gradients...
    """
    def __init__(self, n_split=2):
        super().__init__()

        model_1 = [htmisc.block_selector('convblock2d')(1, 16), htmisc.block_selector('convblock2d')(16, 1)]
        # model_2 = [htmisc.block_selector('convblock2d')(1, 16), htmisc.block_selector('convblock2d')(16, 1)]
        self.n_split = n_split
        # self.seq_model_1 = nn.Sequential(*model_1)
        # self.seq_model_2 = nn.Sequential(*copy.deepcopy(model_1))
        # self.seq_model_3 = [nn.Sequential(*model_1), nn.Sequential(*model_1)]
        # self.seq_model_3 = [self.seq_model_1, self.seq_model_2]
        self.seq_model_3 = nn.ModuleList([nn.Sequential(*model_1) for _ in range(2)])

    def forward(self, x):
        x_split = torch.chunk(x, self.n_split, dim=1)
        res_split = []
        for i in range(self.n_split):
            if i == 0:
                # temp = self.seq_model_1(x_split[0])
                temp = self.seq_model_3[0](x_split[0])
            else:
                # temp = self.seq_model_2(x_split[0])
                temp = self.seq_model_3[1](x_split[1])

            res_split.append(temp)

        res = torch.cat(res_split, dim=1)
        return res


# model_obj = DummyModel().float()
model_obj = DummyModel2().float()
model_params = model_obj.parameters()

htmisc.init_weights(model_obj, 'kaiming')

# Target
B_tens = torch.stack([torch.from_numpy(Z), torch.from_numpy(W)], dim=0)[np.newaxis].float()

import torch.optim
optimizer_obj = torch.optim.Adam(lr=0.01, params=model_params)
loss_obj = torch.nn.MSELoss()

n_epoch = 100
store_grad = []
store_param = []
for _ in range(n_epoch):
    optimizer_obj.zero_grad()
    # Input
    A = np.random.rand(1, 2, len(x), len(y))
    A_tens = torch.from_numpy(A).float()

    res = model_obj(A_tens)

    loss = loss_obj(res, B_tens)
    # loss_0 = loss_obj(res[:, 0], B_tens[:, 0].float())
    # loss_1 = loss_obj(res[:, 1], B_tens[:, 1].float())
    # loss = loss_0 + loss_1
    loss.backward()
    optimizer_obj.step()

    # Now get the gradients
    list_children = htmisc.get_all_children(model_obj, [])
    sel_layer_name, sel_layer_param = htmisc.get_all_parameters(list_children)

    grad_level = htmisc.get_grad_layers(sel_layer_param, sel_layer_name)
    param_level = htmisc.get_param_layers(sel_layer_param, sel_layer_name)
    grad_name, grad_array = zip(*grad_level)
    param_name, param_array = zip(*param_level)
    grad_per_layer = [(float(x.min()), float(x.mean()), float(x.max())) for x in grad_array]
    param_per_layer = [(float(x.min()), float(x.mean()), float(x.max())) for x in param_array]

    store_grad.append(grad_per_layer)
    store_param.append(param_per_layer)
    print('')
    n_space = 30
    for i_name, i in zip(grad_name, param_per_layer):
        temp_min, temp_mean, temp_max = i
        print(i_name, temp_min, (n_space - len(str(temp_min))) * ' ', end='')
        print(temp_mean, (n_space - len(str(temp_mean))) * ' ', end='')
        print(temp_max, (n_space - len(str(temp_max))) * ' ', end='\n')


# Check the final result
res = model_obj(A_tens)
hplotf.plot_3d_list(res.detach().numpy())

# Check the output of gradients
get_grad_fig(store_grad)

# Check the output of parameters
get_grad_fig(store_param)