import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt



n_points = 100
x_range = np.linspace(0, np.pi, n_points)
y_sin = np.sin(x_range)
y_noise = np.random.uniform(0, 1, size=n_points) * 0.1
y_sin_noise = y_sin + y_noise

plt.plot(x_range, y_sin_noise, 'r', label='noisy signal', linewidth=2)
plt.plot(x_range,scipy.ndimage.gaussian_filter(y_sin_noise, 2), 'k', label='smoothed signal', linewidth=2)
plt.legend()


import torch
import torch.nn as nn
import skimage.data
A = skimage.data.astronaut()
A = np.moveaxis(A, -1, 0)
A_tens = torch.from_numpy(A).float()
A = np.array([[1,1,2], [2,3,3], [2, 0 , 1]])
A_tens = torch.from_numpy(A).float()

def _transf(x, n_rep):
    return np.moveaxis(np.tile(x[:, :, None], n_rep), -1, 0)[:, None]

n_channels = 10
B_ones = np.ones((3, 3))
B_eye = np.eye(2)
B_derp = np.array([[2, 0], [1, 0]])
B_zeros = np.zeros((3, 3))
B_diff_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
B_diff_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

B_diff_y_chan = _transf(B_diff_y, n_rep=n_channels)
B_diff_x_chan = _transf(B_diff_x, n_rep=n_channels)
B_ones_chan = _transf(B_ones, n_rep=n_channels)
B_zeros_chan = _transf(B_zeros, n_rep=n_channels)
# Set new conv weights
B_new = np.concatenate([B_ones_chan, B_diff_x_chan, B_diff_y_chan], axis=1)
B_new[1] = np.stack([B_diff_x, B_diff_x, B_diff_x], axis=0)

B_new = np.stack([B_eye, B_derp], axis=0)

conv_layer = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, bias=False, stride=1)

conv_layer.weight = torch.nn.Parameter(torch.from_numpy(B_new[:, None]).float())

with torch.no_grad():
    res = conv_layer(A_tens[None, None])

res
import helper.plot_class as hplotc
hplotc.SlidingPlot(res, cbar=True)
hplotc.SlidingPlot(B_new, cbar=True)



"""
Extractin group/groep numbers from project files
"""

import os
import re
ddata = '/home/bugger/Documents/TUE/AssistentDocent/Assignment1/submissions'
list_dir = os.listdir(ddata)
re_obj = re.compile("(group(\s*|_)|groep(\s*|_))([0-9]+)")
sorted([re_obj.findall(x.lower()) for x in list_dir])
sorted_list = [None] * 20
for x in list_dir:
    re_find = re_obj.findall(x.lower())
    if re_find:
        # print('Group', int(re_find[0][-1]), '\t', x)
        group_index = int(re_find[0][-1])
        if sorted_list[group_index-1] is None:
            sorted_list[group_index-1] = x
        else:
            print("Double group", sorted_list[group_index-1], x)
    else:
        print('Unknown Group', x)

for i, x in enumerate(sorted_list):
    print(f'Group {i+1} ', x)

