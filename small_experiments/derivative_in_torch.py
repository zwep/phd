"""
Nice page about numeric differentiation schemes

https://en.wikipedia.org/wiki/Finite_difference_coefficient

"""

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn
import torch
import helper.plot_fun as hplotf
import helper.plot_class as hplotc

x_range = np.arange(-3, 3, 0.005)
dx = np.diff(x_range)[0]
y = x_range ** 3
y_x = 3 * x_range ** 2

filter = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1, bias=False)
kernel_2 = np.array([-1/2, 0, 1/2])
kernel_4 = np.array([1/12, -2/3, 0, 2/3, -1/12])
kernel_6 = np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
kernel_8 = np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280])

for i, kernel in enumerate([kernel_2, kernel_4, kernel_6, kernel_8]):
    print(i)
    kernel = torch.from_numpy(kernel).view(1, 1, len(kernel))
    filter.weight.data = kernel
    filter.weight.requires_grad = False
    filter.kernel_size = len(kernel)
    y_diff = filter(torch.as_tensor(y[None, None]))
    y_diff = np.pad((1/(dx)) * y_diff[0][0].numpy(), (i, i))
    # y_diff_diff = filter(y_diff)

    plt.plot(y_x - y_diff, alpha=1)
    plt.ylim(-1, 1)


x_range = np.arange(-3, 3, 0.005)
X, Y = np.meshgrid(x_range, x_range)
dx = np.diff(x_range)[0]
Z = np.cos(X) + np.sin(Y)
Z_x = -np.sin(X)
Z_xx = -np.cos(X)
Z_y = np.cos(Y)
Z_yy = -np.sin(Y)

filter = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1, bias=False)
kernel_2 = np.array([-1/2, 0, 1/2])
kernel_4 = np.array([1/12, -2/3, 0, 2/3, -1/12])
kernel_6 = np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
kernel_8 = np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280])

for i, kernel in enumerate([kernel_2, kernel_4, kernel_6, kernel_8]):
    n_kernel = len(kernel)
    kernel = np.tile(kernel, n_kernel).reshape(n_kernel, -1).T
    kernel = torch.from_numpy(kernel).view(1, 1, n_kernel, n_kernel).float()
    filter.weight.data = kernel
    filter.weight.requires_grad = False
    filter.kernel_size = n_kernel
    # filter.stride = tuple(np.ones(n_kernel, dtype=int))
    y_diff = filter(torch.as_tensor(Z[None, None]).float())
    y_diff = np.pad((1/(dx)) * y_diff[0][0].numpy(), (i, i))
    # y_diff_diff = filter(y_diff)

    hplotf.plot_3d_list([Z_y, 1/n_kernel * y_diff], vmin=(-1,1))

"""
Second order derivative....
"""

filter = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1, bias=False)
kernel_2 = np.array([1, -2, 1])
kernel_4 = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])
kernel_6 = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
kernel_8 = np.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560])


for i, kernel in enumerate([kernel_2, kernel_4, kernel_6, kernel_8]):
    n_kernel = len(kernel)
    kernel = np.tile(kernel.T, n_kernel).reshape(n_kernel, -1)
    kernel = torch.from_numpy(kernel).view(1, 1, n_kernel, n_kernel).float()
    filter.weight.data = kernel
    filter.weight.requires_grad = False
    filter.kernel_size = n_kernel
    # filter.stride = tuple(np.ones(n_kernel, dtype=int))
    y_diff = filter(torch.as_tensor(Z[None, None]).float())
    y_diff = np.pad((1/(dx) ** 2) * y_diff[0][0].numpy(), (i, i))
    # y_diff_diff = filter(y_diff)

    hplotf.plot_3d_list([Z_xx, 1/n_kernel * y_diff], vmin=(-1,1))
