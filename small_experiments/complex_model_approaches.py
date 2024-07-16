
"""
Here we test
"""

import helper_torch.layers as hlayer
import model.UNet as UNet
import matplotlib
import matplotlib.pyplot as plt

import torch.nn
import torch_optimizer as optim
import model.Blocks as Blocks
import helper_torch.misc as htmisc
import helper_torch.loss as hloss

import numpy as np
import helper.array_transf as harray
import torch
import helper.plot_fun as hplotf
import helper.plot_class as hplotc



i_file = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels/train/input/11_20191023_03.npy'
input_array = np.load(i_file)
input = input_array.sum(axis=0)
tgt = input_array.sum(axis=1)
input_line = input[0][256, :]
tgt_line = tgt[0][256, :]

N = 50
x_orig = np.outer(np.linspace(-2, 2, N), np.ones(N))
y_orig = x_orig.copy().T  # transpose
# z0 = np.exp(- ((1.5*x) ** 2 + (1.2*y) ** 2)) * np.exp(- 1j * np.sin(1 * (x ** 2 + (0.8*y) ** 2)))
# z1 = np.exp(- ((0.5*x) ** 2 + (1.2*y) ** 2)) * np.exp(- 1j * np.sin(3 * (x ** 2 + (0.8*y) ** 2)))
z0 = np.exp(- (1.0 * x_orig ** 2 + 1.0 * y_orig ** 2)) * np.exp(1j * np.sin(2 * (x_orig ** 2 + y_orig ** 2)))
z1 = np.log(z0)

for iaugm in ['np.real', 'np.imag', 'np.abs', 'np.angle']:
    hplotf.plot_3d_list([z0, z1], augm=iaugm, title=iaugm)

hplotc.SlidingPlot(z0, ax_3d=True)
hplotc.SlidingPlot(z1, ax_3d=True)
hplotc.SlidingPlot(z0/z1, ax_3d=True)

# Transform objects
import helper_torch.transforms as htransform

t_std = htransform.TransformStandardize(prob=False)

# Complex fft
# z0 = harray.transform_image_to_kspace_fftn(z0)
# z1 = harray.transform_image_to_kspace_fftn(z1)
# Complex normal
z0_stack = harray.to_stacked(z0, stack_ax=0)[np.newaxis]
z0_stack = t_std(z0_stack, ax=(-2, -1))

z1_stack = harray.to_stacked(z1, stack_ax=0)[np.newaxis]
z1_stack = t_std(z1_stack, ax=(-2, -1))

# Complex..
z0_tens = torch.as_tensor(z0_stack).float()
z1_tens = torch.as_tensor(z1_stack).float()
# Abs part..
# z0_tens = torch.as_tensor(np.abs(z0)).float()
# z1_tens = torch.as_tensor(np.abs(z1)).float()
# Real part..
# z0_tens = torch.as_tensor(np.real(z0)).float()
# z1_tens = torch.as_tensor(np.real(z1)).float()

# Create some batch like things...
z0_tens = z0_tens.repeat((16, 1, 1, 1))
z1_tens = z1_tens.repeat((16, 1, 1, 1))


import model.Basic as Basic

mod_obj = UNet.UnetModel(in_chans=2, out_chans=2, chans=8, drop_prob=0.1, num_pool_layers=2)
# mod_obj = Basic.SimpleModelConv2D(in_chan=2, start_chan=16, out_chan=2, n_layer=2, actv='relu', group_list=1)
# mod_obj = Basic.SimpleModelConv2D(in_chan=2, start_chan=8, out_chan=2, n_layer=3, actv='relu', group_list=[2, 2, 1])
# mod_obj = Basic.DenseModel(ndim_start=N, ndim_hidden=[60, 70, 80], ndim_out=N, actv='sigmoid')
# mod_obj = DenseModelV2(ndim_start=N, ndim_hidden=[30, 40], ndim_out=N, actv='sigmoid')

with torch.no_grad():
    res = mod_obj(z0_tens)

sel_batch = 0
hplotf.plot_3d_list(res[sel_batch], title='initial plot')

optim_obj = optim.adabound.AdaBound(lr=0.001, params=mod_obj.parameters())

loss_abs = hloss.AbsL1Loss()
loss_angle = hloss.AngleL1Loss()
loss_fft = hloss.FFTL1Loss()
loss_real_imag = torch.nn.L1Loss()
# loss_obj_list = [loss_abs, loss_angle, loss_fft, loss_real_imag]
# loss_obj_list = [loss_abs, loss_real_imag]
loss_obj_list = [loss_real_imag]

# plt.close('all')
eps = 0.0001
counter = 0
loss_item = 100
loss_item_prev = 0
mod_obj.train()
loss_list = []
break_down = 0

grad_norm_time = []
grad_filter_time = []
n_breakdown = 10
while break_down < n_breakdown and counter < 2500:
    loss_item_prev = loss_item
    optim_obj.zero_grad()
    z_pred = mod_obj(z0_tens)

    loss_index = counter % len(loss_obj_list)
    loss = loss_obj_list[loss_index](z_pred, z1_tens)
    loss.backward()

    if counter % 2 == 0:
        param_list = list(mod_obj.parameters())
        param_list = [x for x in param_list if x.ndim > 1]
        # grad_list = [x.grad for x in param_list]  #
        grad_list = []
        for x in param_list:
            temp_shape = (-1,) + x.shape[-2:]
            x_temp = x.grad.reshape(temp_shape)
            grad_list.append(x_temp)
        grad_norm = [x.norm(p=2, dim=(-2, -1)) for x in grad_list]
        grad_norm_time.append(grad_norm)

    if counter % 10 == 0:
        grad_filtered = []
        for x in grad_list:
            temp_max_ind = np.argmax(x.norm(p=2, dim=(-2, -1)))
            temp_min_ind = np.argmin(x.norm(p=2, dim=(-2, -1)))
            temp_tens = torch.stack([x[temp_max_ind], x[temp_min_ind]])
            grad_filtered.append(temp_tens)
        grad_filter_time.append(grad_filtered)

    optim_obj.step()

    loss_item = loss.item()
    loss_list.append(loss_item)

    criterion = np.mean(loss_list[-20:-10]) - np.mean(loss_list[-10:])
    if criterion < 0:
        break_down += 1
    else:
        break_down -= 1
        break_down = max(break_down, 0)

    counter += 1
    if counter % int(0.1 * 2500) == 0:
        print(counter, criterion, break_down, loss_item)

print('finalcounter', counter)

plot_output = True

if plot_output:

    n_fig = len(grad_norm_time[0])
    test = [[] for _ in range(n_fig)]
    for i_time in range(len(grad_norm_time)):
        for i_fig in range(n_fig):
            temp = np.array(grad_norm_time[i_time][i_fig])
            test[i_fig].append(temp)


    import helper.misc as hmisc
    fig, ax_list = plt.subplots(n_fig)
    ax_list = ax_list.ravel()
    cmap_col = 'Reds'
    plt_cm = plt.get_cmap(cmap_col)

    # Needed for conv layers...
    conv_model = True

    # Needed for linear layers...
    if not conv_model:
        v_max = np.max(test)
        for j in range(n_fig):
            plot_test = test[j]
            ax_list[j].plot(plot_test)
            ax_list[j].set_ylim(0, v_max)
    else:
        for j in range(n_fig):
            test_2 = hmisc.change_list_order(test[j])
            num_lines = len(test_2)
            color_list = [plt_cm(1. * i / num_lines) for i in range(num_lines)]
            ax_list[j].set_prop_cycle('color', color_list)
            for i in test_2:
                ax_list[j].plot(i)
            ax_list[j].set_ylim(0, 0.5)

    plt.figure()
    plt.plot(loss_list[::2])

with torch.no_grad():
    z_final = mod_obj(z0_tens).numpy()[0]

hplotf.plot_3d_list(z_final, title='outcome')
hplotf.plot_3d_list(z_final - z1_tens.numpy()[0], title='outcome')

hplotc.SlidingPlot(z_final, title='outcome', ax_3d=True)
hplotc.SlidingPlot(z_final - z1_tens[0].numpy(), title='target', ax_3d=True)

hplotf.plot_3d_list(z1_tens[0], title='target')
hplotf.plot_3d_list(z0_tens[0], title='input')


random_tens = torch.as_tensor(np.random.rand(10, 1, N, N)).float()
with torch.no_grad():
    z_final_random = mod_obj(random_tens).numpy()

hplotf.plot_3d_list(z_final_random, title='outcome random')


hplotc.SlidingPlot(z1_tens[0].numpy() - z_final, ax_3d=True)
hplotc.SlidingPlot(z_final, ax_3d=True)
hplotc.SlidingPlot(z1_tens[0], ax_3d=True)

"""
Showing intermediate stuff....
"""

res = htmisc.get_all_children(mod_obj, [])

n = None
interm_layers = []
with torch.no_grad():
    x = z0_tens[sel_batch:sel_batch+1]
    for i_layer in res[:n]:
        x = i_layer(x)
        interm_layers.append(x)

proc_layers = []
for x in interm_layers:
    norm_channels = x.norm(p=2, dim=-1).norm(p=2, dim=-1)
    ind_chan = np.argmax(norm_channels, axis=1)
    proc_layers.append(x[0, ind_chan[0]].numpy())

hplotf.plot_3d_list(np.array(proc_layers)[np.newaxis])

"""
Complex model with Torch...?

"""

import torch
import torch.nn as nn
import torch.optim as toptim


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
A1 = np.random.rand(1, 10, 10)
B_target = np.eye(10)[np.newaxis]
B_tens = torch.as_tensor(B_target).float()
A_tens = torch.as_tensor(A + 1j * A1, dtype=torch.complex64)
result1 = model_obj(A_tens)
result2 = model_obj(torch.as_tensor(A).float())
result1==result2