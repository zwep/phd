
import torch
import torch.nn as nn
import torch.optim as toptim
import helper_torch.misc as htmisc
import numpy as np
import helper_torch.misc as hisc
import helper.plot_fun
import matplotlib.pyplot as plt

"""
From this we can see that using the detach and afterwards a loss does indeed add to the gradients

WRONG!

I forgot to re-assign the tensor. So it does NOT add to any gradient calculation...
"""


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(10, 20),
            nn.Sigmoid(),
            nn.Linear(20, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

def transform_output(x_tens):
    x_tens = x_tens.detach()
    return torch.sin(x_tens) ** 3

loss_obj = nn.L1Loss()
loss_obj2 = nn.L1Loss()
model_obj = SimpleModel()
optim_obj = toptim.Adam(model_obj.parameters())

torch.manual_seed(0)
A = np.ones((1, 10, 10))
B_target = np.eye(10)[np.newaxis]
B_tens = torch.as_tensor(B_target).float()
A_tens = torch.as_tensor(A).float()


result = model_obj(A_tens)
loss_item = loss_obj(result, B_tens)

result_trans = transform_output(result)
loss_item2 = loss_obj2(result_trans, B_tens)

# loss_item.backward(retain_graph=True)
loss_item2.backward()
# loss_item += loss_item2
# loss_item.backward()


list_children = htmisc.get_all_children(model_obj, [])
sel_layer_name, sel_layer_param = htmisc.get_all_parameters(list_children)
grad_level = htmisc.get_grad_layers(sel_layer_param, sel_layer_name)
grad_level_list = [float(x[1]) for x in grad_level]
print(grad_level_list)
optim_obj.zero_grad()
