# encoding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
import torch.nn
import itertools
import torch.nn.functional as F


def instance_std(x, eps=1e-5):
    var = torch.var(x, dim=(2, 3), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)


def group_std(x, groups=1, eps=1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, max(C // groups, 1), H, W))
    var = torch.var(x, dim=(2, 3, 4), keepdim=True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))


class EvoNorm2D(torch.nn.Module):
    # source https://github.com/digantamisra98/EvoNorm
    def __init__(self, input, non_linear=True, version='S0', affine=True,
                 momentum=0.9, eps=1e-5, training=True, groups=1):
        super(EvoNorm2D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.eps = eps
        self.groups = groups
        if self.version not in ['B0', 'S0']:
            raise ValueError("Invalid EvoNorm version")

        self.insize = input
        self.affine = affine

        if self.affine:
            self.gamma = torch.nn.Parameter(torch.ones(1, self.insize, 1, 1), requires_grad=True)
            self.beta = torch.nn.Parameter(torch.zeros(1, self.insize, 1, 1), requires_grad=True)
            if self.non_linear:
                self.v = torch.nn.Parameter(torch.ones(1, self.insize, 1, 1), requires_grad=True)
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
            self.register_buffer('v', None)
        self.register_buffer('running_var', torch.ones(1, self.insize, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.version == 'S0':
            if self.non_linear:
                num = x * torch.sigmoid(self.v * x)
                return num / group_std(x, eps=self.eps, groups=self.groups) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == 'B0':
            if self.training:
                var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var)
            else:
                var = self.running_var

            if self.non_linear:
                den = torch.max((var + self.eps).sqrt(), self.v * x + instance_std(x, eps=self.eps))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta


# Rank-ratio version
class xCNNrankratio(torch.nn.Module):
    # https://github.com/kumarak93/LinearConv
    def __init__(self, in_chans, out_chans, kernel_size, padding=1, stride=1, groups=1, rank=1, bias=True, **kwargs):
        super().__init__()
        self.out_chans = out_chans
        self.times = 2
        self.out_chans_times = max(1, out_chans // self.times)
        self.kernel_size = kernel_size
        self.in_chans = in_chans // groups
        self.padding = padding
        self.stride = stride
        self.biasTrue = bias
        self.rank = rank
        self.groups = groups
        self.device = kwargs.get('device', 'cpu')

        conv_tens = torch.Tensor(self.out_chans_times, in_chans, kernel_size, kernel_size).to(self.device)
        self.xcnn_weights = torch.nn.Parameter(conv_tens, requires_grad=True)
        column_tens = torch.Tensor(out_chans - self.out_chans_times, int(self.out_chans_times * self.rank)).to(self.device)
        self.column_weights = torch.nn.Parameter(column_tens, requires_grad=True)
        row_tens = torch.Tensor(int(self.out_chans_times * self.rank), self.out_chans_times).to(self.device)
        self.row_weights = torch.nn.Parameter(row_tens, requires_grad=True)

        torch.nn.init.xavier_uniform_(self.xcnn_weights)
        self.column_weights.data.uniform_(-0.1, 0.1)
        self.row_weights.data.uniform_(-0.1, 0.1)

        if self.biasTrue:
            bias_tens = torch.Tensor(out_chans).to(self.device)
            self.bias = torch.nn.Parameter(bias_tens, requires_grad=True)
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        temp = torch.mm(self.row_weights, self.xcnn_weights.reshape(self.out_chans_times, -1))
        self.correlated_weights = torch.mm(self.column_weights, temp)
        new_shape = self.out_chans - self.out_chans_times, self.in_chans, self.kernel_size, self.kernel_size
        self.correlated_weights = self.correlated_weights.reshape(new_shape)

        cat_weights = torch.cat((self.xcnn_weights, self.correlated_weights), dim=0)
        if self.biasTrue:
            output = F.conv2d(input, cat_weights, bias=self.bias, padding=self.padding, stride=self.stride)
        else:
            output = F.conv2d(input, cat_weights, padding=self.padding, stride=self.stride)

        return output


# Const. low-rank version
class xCNNlowrank(torch.nn.Module):
    # https://github.com/kumarak93/LinearConv
    def __init__(self, in_chans, out_chans, kernel_size, padding=0, stride=1, groups=1, rank=1, bias=True, **kwargs):
        super().__init__()
        # self.device = kwargs.get('device', 'cpu')
        self.out_chans = out_chans
        self.times = 2
        self.out_chans_times = max(1, out_chans//self.times)
        self.kernel_size = kernel_size
        self.in_chans = in_chans // groups
        self.padding = padding
        self.stride = stride
        self.biasTrue = bias
        self.rank = rank
        self.groups = groups

        self.xcnn_weights = torch.nn.Parameter(
            torch.Tensor(self.out_chans_times, in_chans, kernel_size, kernel_size),
            requires_grad=True)
        self.column_weights = torch.nn.Parameter(torch.Tensor(out_chans - self.out_chans_times, self.rank),
                                                 requires_grad=True)
        self.row_weights = torch.nn.Parameter(torch.Tensor(self.rank, self.out_chans_times),
                                              requires_grad=True)
        # print('CNN low weight shape', self.xcnn_weights.shape)
        torch.nn.init.xavier_uniform_(self.xcnn_weights)
        self.column_weights.data.uniform_(-0.1, 0.1)
        self.row_weights.data.uniform_(-0.1, 0.1)

        if self.biasTrue:
            self.bias = torch.nn.Parameter(torch.Tensor(out_chans), requires_grad=True)
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        self.correlated_weights = torch.mm(self.column_weights, torch.mm(self.row_weights, self.xcnn_weights.reshape(
            self.out_chans_times, -1))) \
            .reshape(self.out_chans - self.out_chans_times, self.in_chans, self.kernel_size, self.kernel_size)
        if self.biasTrue:
            return F.conv2d(input, torch.cat((self.xcnn_weights, self.correlated_weights), dim=0), \
                            bias=self.bias, padding=self.padding, stride=self.stride)
        else:
            return F.conv2d(input, torch.cat((self.xcnn_weights, self.correlated_weights), dim=0), \
                            padding=self.padding, stride=self.stride)


# Sparse version
class xCNNsparse(torch.nn.Module):
    # https://github.com/kumarak93/LinearConv
    def __init__(self, in_chans, out_chans, kernel_size, padding=1, stride=1, groups=1, bias=True, **kwargs):
        super().__init__()
        self.out_chans = out_chans
        self.times = 2  # ratio 1/2
        self.kernel_size = kernel_size
        self.out_chans_times = max(1, out_chans // self.times)
        self.in_chans = in_chans // groups
        self.padding = padding
        self.stride = stride
        self.biasTrue = bias
        self.groups = groups
        self.device = kwargs.get('device', 'cpu')
        self.counter = 0
        self.threshold = 0
        # Default value of prune_step, req_percentile, and thres_step
        # https://github.com/kumarak93/LinearConv/blob/master/models/base_xcnn.py
        self.prune_step = kwargs.get('prune_step', 500)
        self.req_percentile = kwargs.get('req_percentile', 0.25)
        self.thres_step = kwargs.get('thres_step', 0.00001) #0.001
        # self.mask = torch.abs(self.linear_weights) > self.threshold

        self.xcnn_weights = torch.nn.Parameter(torch.Tensor(self.out_chans_times, in_chans, kernel_size, kernel_size), requires_grad=True)
        self.linear_weights = torch.nn.Parameter(torch.Tensor(out_chans - self.out_chans_times, self.out_chans_times), requires_grad=True)

        torch.nn.init.xavier_uniform_(self.xcnn_weights)
        self.linear_weights.data.uniform_(-0.1, 0.1)

        # self.mask = torch.abs(self.linear_weights) > self.threshold

        if self.biasTrue:
            self.bias = torch.nn.Parameter(torch.Tensor(out_chans), requires_grad=True)
            self.bias.data.uniform_(-0.1, 0.1)

        self.mask = torch.nn.Parameter(torch.abs(self.linear_weights) > self.threshold, requires_grad=False)
        self.mask = self.mask.float()
        # self.mask = self.mask.to(self.device)


    def forward(self, input):
        self.counter += 1
        if self.counter == self.prune_step:
            self.counter = 0
            self.mask = torch.nn.Parameter(torch.abs(self.linear_weights) > self.threshold, requires_grad=False)
            self.percentile = 1. - float(torch.sum(self.mask).item()) / (self.mask.shape[0] ** 2)
            # self.threshold += (req_percentile - self.percentile) * thres_step
            self.threshold += (2. / (1. + 10 ** (10 * (self.percentile - self.req_percentile))) - 1) * self.thres_step
            print('pruned... %.2f, %.5f' % (self.percentile, self.threshold))

        # self.mask = torch.nn.Parameter(self.mask.type(torch.FloatTensor).to(self.device), requires_grad=False)
        temp = self.linear_weights * self.mask
        self.correlated_weights = torch.mm(temp, self.xcnn_weights.reshape(self.out_chans_times, -1)) \
            .reshape(self.out_chans - self.out_chans_times, self.in_chans, self.kernel_size, self.kernel_size)

        if self.biasTrue:
            return F.conv2d(input, torch.cat((self.xcnn_weights, self.correlated_weights), dim=0), \
                            bias=self.bias, padding=self.padding, stride=self.stride)
        else:
            return F.conv2d(input, torch.cat((self.xcnn_weights, self.correlated_weights), dim=0), \
                            padding=self.padding, stride=self.stride)


class xCNN(torch.nn.Module):
    # https://github.com/kumarak93/LinearConv
    def __init__(self, in_chans, out_chans, kernel_size, padding=0, stride=1, groups=1, bias=True, **kwargs):
        super().__init__()
        self.out_chans = out_chans
        self.device = kwargs.get('device', 'cpu')
        self.times = 2  # ratio 1/2
        self.kernel_size = kernel_size
        self.out_chans_times = max(1, out_chans // self.times)
        self.in_chans = in_chans // groups
        self.padding = padding
        self.stride = stride
        self.biasTrue = bias
        self.groups = groups

        self.xcnn_weights = torch.nn.Parameter(
            torch.Tensor(self.out_chans_times, in_chans, kernel_size, kernel_size).to(self.device),
            requires_grad=True)
        self.linear_weights = torch.nn.Parameter(
            torch.Tensor(out_chans - self.out_chans_times, self.out_chans_times).to(self.device),
        requires_grad=True)

        torch.nn.init.xavier_uniform_(self.xcnn_weights)
        self.linear_weights.data.uniform_(-0.1, 0.1)

        if self.biasTrue:
            self.bias = torch.nn.Parameter(torch.Tensor(out_chans).to(self.device),
                                           requires_grad=True)
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        self.correlated_weights = torch.mm(self.linear_weights,
                                           self.xcnn_weights.reshape(self.out_chans_times, -1)) \
            .reshape(self.out_chans - self.out_chans_times, self.in_chans, self.kernel_size, self.kernel_size)

        if self.biasTrue:
            return F.conv2d(input, torch.cat((self.xcnn_weights, self.correlated_weights), dim=0), \
                            bias=self.bias, padding=self.padding, stride=self.stride)
        else:
            return F.conv2d(input, torch.cat((self.xcnn_weights, self.correlated_weights), dim=0), \
                            padding=self.padding, stride=self.stride)


class LearnableRotation(torch.nn.Module):
    """
    Learnable rotation matrix
    """
    def __init__(self):
        super().__init__()
        temp_theta = torch.as_tensor(np.random.uniform(low=-np.pi, high=np.pi, size=1))
        self.theta = torch.nn.Parameter(temp_theta, requires_grad=True)

    def forward(self, x):
        rot_tens = torch.Tensor([[torch.cos(self.theta), -torch.sin(self.theta)],
                                 [torch.sin(self.theta), torch.cos(self.theta)]]).float()
        rot_tens = rot_tens.to(self.theta.device)
        res_rot = torch.einsum("...n, nm->...m", x, rot_tens)
        return res_rot


class RotationLayer(torch.nn.Module):
    """
    NOT-Learnable rotation matrix. Tested. Works as intendend
    """
    def __init__(self, N=30):
        super().__init__()
        temp_theta = torch.Tensor(np.linspace(0, 360, N) * np.pi/180)
        self.theta_range = torch.nn.Parameter(temp_theta, requires_grad=False)

    def forward(self, x):
        rot_vect = [torch.cos(self.theta_range), -torch.sin(self.theta_range),
                    torch.sin(self.theta_range), torch.cos(self.theta_range)]
        rot_tens = torch.stack(rot_vect, dim=0).reshape((2, 2, len(self.theta_range))).float()
        rot_tens = rot_tens.to(self.theta_range.device)
        res_rot = torch.einsum("...n, nmz->...mz", x, rot_tens)
        return res_rot



class HighFrequentyLayer(torch.nn.Module):
    """ Perform positional encoding.

    A data augmentation trick to transform the input to a high freq version.
    """
    def __init__(self, L):
        super().__init__()
        self.L = L
        position_enc = [(np.pi * 2 ** i, np.pi * 2 ** i) for i in range(L)]
        position_enc_tens = torch.as_tensor(np.array(list(itertools.chain(*position_enc))))
        position_enc_tens = position_enc_tens.reshape(1, 2 * L).float()
        self.position_encoding = torch.nn.Parameter(position_enc_tens, requires_grad=False)

    def forward(self, x):
        # Assuming input being of the form (batch, channel, ...)
        n_chan = x.shape[1]
        n_chan_new = n_chan * 2 * self.L

        # Multiply by pi 2 ** i
        res = [torch.einsum("bcmn, cl -> blmn", x[:, i:i+1], self.position_encoding) for i in range(n_chan)]
        res_tens = torch.cat(res, dim=1)  # Concat on channel dimension

        # Now apply sin/cos for each channel in an alternating fashion
        res_cosin = [torch.sin(res_tens[:, i:i + 1]) if i % 2 == 0 else torch.cos(res_tens[:, i:i + 1]) for i in
                     range(n_chan_new)]
        res_cosin_tens = torch.cat(res_cosin, dim=1)  # Concat on channel dimension
        return res_cosin_tens


class SplitPosNegLayer(torch.nn.Module):
    # Assumes input of (?, 2, Y, X)
    # output of (?, 4, Y, X)
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = torch.stack([self.relu(x[:, 0]), self.relu(-x[:, 0]), self.relu(x[:, 1]), self.relu(-x[:, 1])], dim=1)
        return x


class SplitCatLayer(torch.nn.Module):
    # See if this one has any use...
    def __init__(self, dim_split, dim_cat, size_split):
        super().__init__()
        self.dim_split = dim_split
        self.dim_cat = dim_cat
        self.size_split = size_split

    def forward(self, x):
        x_list = x.split(self.size_split, dim=self.dim_split)
        x = torch.cat(x_list, dim=self.dim_cat)
        return x


class SplitStackLayer(torch.nn.Module):
    # See if this one has any use...
    def __init__(self, dim_split, dim_cat, size_split):
        super().__init__()
        self.dim_split = dim_split
        self.dim_cat = dim_cat
        self.size_split = size_split

    def forward(self, x):
        x_list = x.split(self.size_split, dim=self.dim_split)
        x = torch.stack(x_list, dim=self.dim_cat)
        return x


class Transpose2D(torch.nn.Module):
    # Swaps (B, C, H, W) <-> (B, H, W, C)
    # to_channel_last: True ->
    # to_channel_last: False <-
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.permute((0, 1, 3, 2))
        return x


class SwapAxes2D(torch.nn.Module):
    # Swaps (B, C, H, W) <-> (B, H, W, C)
    # to_channel_last: True ->
    # to_channel_last: False <-
    def __init__(self, to_channel_last=True, requires_grad=False):
        super().__init__()
        self.to_channel_last = to_channel_last
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.to_channel_last:
            x = x.permute((0, 2, 3, 1))
        else:
            x = x.permute((0, 3, 1, 2))

        return x


class SwapAxes3D_special(torch.nn.Module):
    # Swaps (B, C, C_cpx, H, W) <-> (B, C, H, W, C_cpx)
    # to_channel_last: True ->
    # to_channel_last: False <-
    # For now this is used to move the complex-channel axes to the last, so that we can use it in a linear layer
    # This was created so that, if we use .get_children_layers, we avoid any manual/implicit function tasks in the model.
    def __init__(self, to_channel_last=True, requires_grad=False):
        super().__init__()
        self.to_channel_last = to_channel_last
        # There are no parameters here.. I know..
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.to_channel_last:
            x = x.permute((0, 1, 3, 4, 2))
        else:
            x = x.permute((0, 1, 4, 2, 3))

        return x


class SinkhornLayer(torch.nn.Module):
    # https://github.com/alexeyev/SinkhornLayer-keras
    # What kind of input shape to expect...
    def __init__(self, n_iters=21, temperature=0.04):
        super().__init__()
        self.n_iters = n_iters
        self.temperature = temperature

    def forward(self, input_tensor):
        n = input_tensor.shape[1]
        log_alpha = input_tensor.reshape((-1, n, n))
        log_alpha /= self.temperature

        for _ in range(self.n_iters):
            log_alpha -= torch.logsumexp(log_alpha, dim=2).reshape((-1, n, 1))
            log_alpha -= torch.logsumexp(log_alpha, dim=1).reshape((-1, 1, n))

        return torch.exp(log_alpha)


if __name__ == "__main__":
    import numpy as np
    A = np.random.rand(1, 3, 10, 12)
    A_tens = torch.as_tensor(A).float()
    swap_ax_layer_to = SwapAxes2D(to_channel_last=True)
    swap_ax_layer_from = SwapAxes2D(to_channel_last=False)
    print('initial_shape', A_tens.shape)
    out_tens = swap_ax_layer_to(A_tens)
    print('swaped shape', out_tens.shape)
    orig_tens = swap_ax_layer_from(out_tens)
    print('swaped shape', orig_tens.shape)

    # Sinkhorn Layer
    import numpy as np
    A = SinkhornLayer()
    B = np.random.rand(10, 10)
    C = A(torch.as_tensor(B))
    plt.imshow(B)
    plt.imshow(C[0])

    #
    import numpy as np
    import torch
    A_input = np.random.rand(1, 3, 10, 12)
    A_layer = HighFrequentyLayer(3).float()
    res = A_layer(torch.as_tensor(A_input).float())
    hplotc.SlidingPlot(res)
    
    # # # Testing
    L = 5
    import torch
    import itertools
    import numpy as np

    import data_generator.UndersampledRecon as data_gen
    import importlib
    import helper.array_transf as harray
    import helper.plot_fun as hplotf

    importlib.reload(data_gen)
    dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation'
    A = data_gen.DataGeneratorSemireal(dir_data, transform_type='complex', complex_type='polar', shuffle=False)
    A_coarse = data_gen.DataGeneratorSemireal(dir_data, transform_type='complex', complex_type='polar',
                                              image_mode='coarse', shuffle=False, coarse_factor=0.505)
    a_coarse, b_coarse = A_coarse.__getitem__(1)
    A_tens = a_coarse[np.newaxis]
    n_chan = A_tens.shape[1]
    # A = np.random.rand(5, 3, 10, 10)
    # A = np.ones((5, 3, 10, 10))
    # A_tens = torch.as_tensor(A)


    A = torch.as_tensor(np.random.rand(2, 15, 14)).float()

    res = xCNNlowrank(in_chans=2, out_chans=3, kernel_size=3)
    resA = res(A[np.newaxis])
    plt.figure()
    plt.imshow(resA[0, 0].detach().numpy())
    res = xCNNrankratio(in_chans=2, out_chans=3, kernel_size=3)
    resA = res(A[np.newaxis])
    plt.figure()
    plt.imshow(resA[0, 0].detach().numpy())
    res = xCNN(in_chans=2, out_chans=3, kernel_size=3)
    resA = res(A[np.newaxis])
    plt.figure()
    plt.imshow(resA[0, 0].detach().numpy())
    res = xCNNsparse(in_chans=2, out_chans=3, kernel_size=3)
    resA = res(A[np.newaxis])
    plt.figure()
    plt.imshow(resA[0, 0].detach().numpy())
