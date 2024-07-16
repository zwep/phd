import torch
import numpy as np
import helper.plot_class as hplotc
from model.ResNet import ResnetGenerator
import helper.misc as hmisc
import matplotlib.pyplot as plt

import helper_torch.misc as htmisc

def get_param_plot(model_obj):
    mod_children = htmisc.get_all_children(model_obj)
    sel_layer_name, sel_layer_param = htmisc.get_all_parameters(mod_children)
    param_level = htmisc.get_param_layers(sel_layer_param, sel_layer_name)
    param_name, param_array = zip(*param_level)
    param_per_layer = [(float(x.min()), float(x.mean()), float(x.max())) for x in param_array]
    fig_obj = plot_stuff([param_per_layer])
    return fig_obj, param_name

# # #
def plot_stuff(x):
    # Only useful for ONE iteration of the layers..
    x_switch = hmisc.change_list_order(x)
    n_layers = len(x_switch)

    fig, ax = plt.subplots(1, n_layers, figsize=(20, 15))
    # If we have multiple axes... siwtch them..
    if hasattr(ax, 'ravel'):
        ax = ax.ravel()
    else:
        ax = [ax]

    temp_max_y = np.max(x_switch)

    if np.isfinite(temp_max_y):
        temp_max_y += 0.05 * temp_max_y
    else:
        temp_max_y = 1

    for i in range(n_layers):
        temp_min, temp_mean, temp_max = zip(*x_switch[i])
        ax[i].scatter(0, temp_min, color='b', alpha=0.5)
        ax[i].scatter(0, temp_mean, color='k')
        ax[i].scatter(0, temp_max, color='r', alpha=0.5)
        # USe a global maxmimum
        ax[i].set_ylim(0, temp_max_y)
        # OR a local maximum. For now this is desired I guess...
        # Filter out the not finite elements.
        # plot_max = 1.2 * np.max([x for x in temp_max if np.isfinite(x)])
        # ax[i].set_ylim(0, plot_max)

    return fig

"""
Check if weight clipping really works

Conclussion: It does really work
"""


mod_obj = ResnetGenerator(2, 1, downsampling=3)
for i_init in ['normal', 'xavier', 'kaiming', 'orthogonal']:
    htmisc.init_weights(mod_obj, i_init)
    fig_handle, layer_names = get_param_plot(mod_obj)
    fig_handle.suptitle(i_init)

i_init = 'kaiming'
htmisc.init_weights(mod_obj, i_init)
fig_handle, layer_names = get_param_plot(mod_obj)
# Check Resnet parameter clipping...

counter = 0
for p in mod_obj.parameters():
    print(counter, p.data.shape, p.data.min(), p.data.max())
    counter += 1

counter = 0
for p in mod_obj.parameters():
    p.data.clamp_(-0.5, 0.5)

counter = 0
for p in mod_obj.parameters():
    print(counter, p.data.shape, p.data.min(), p.data.max())
    counter += 1

fig_handle = get_param_plot(mod_obj)

mod_children = htmisc.get_all_children(mod_obj)
sel_layer_name, sel_layer_param = htmisc.get_all_parameters(mod_children)
hplotc.SlidingPlot(sel_layer_param[0][0].detach())
param_level = htmisc.get_param_layers(sel_layer_param, sel_layer_name)
param_name, param_array = zip(*param_level)