
from model.VGG import Vgg16
from helper_torch.misc import flatten, extract_patches
import orthopoly
import torch
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import helper_torch.layers as hlayer
import numpy as np
import scipy.special
import torch.nn.functional as F

"""
Physics based losses
"""


class HelmholtzLossEpsilonSigma(torch.nn.Module):
    def __init__(self, dx=1/256, kernel='6'):
        super().__init__()
        # Second order derivatives
        # Taken from wikipedia..
        kernel_2 = np.array([1, -2, 1])
        kernel_4 = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])
        kernel_6 = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
        kernel_8 = np.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560])
        kernel_dict = {'2': [kernel_2, 0], '4': [kernel_4, 1], '6': [kernel_6, 2], '8': [kernel_8, 3]}

        kernel, self.pad = kernel_dict[kernel]
        self.dx = dx
        n_kernel = len(kernel)

        kernel_xx = np.tile(kernel, n_kernel).reshape(n_kernel, -1)
        kernel_xx = torch.nn.Parameter(torch.from_numpy(kernel_xx).view(1, 1, n_kernel, n_kernel).float(), requires_grad=False)
        filter_xx = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1, bias=False)
        filter_xx.weight.data = kernel_xx
        filter_xx.weight.requires_grad = False
        filter_xx.kernel_size = n_kernel

        kernel_yy = np.tile(kernel.T, n_kernel).reshape(n_kernel, -1)
        kernel_yy = torch.nn.Parameter(torch.from_numpy(kernel_yy).view(1, 1, n_kernel, n_kernel).float(), requires_grad=False)
        filter_yy = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1, bias=False)
        # Taking the transpose to transform it to a proper y-directional kernel
        filter_yy.weight.data = kernel_yy
        filter_yy.weight.requires_grad = False
        filter_yy.kernel_size = n_kernel

        self.operator_xx = filter_xx
        self.operator_yy = filter_yy

        self.loss_obj = torch.nn.MSELoss()

    def _pad(self, x):
        return F.pad(x, (self.pad, self.pad, self.pad, self.pad))

    def forward(self, pred, target, x_input):
        # The input variable contains
        # I have not checked the units..
        omega = 42  # ??
        mu_0 = 1e-4  # ??
        epsilon = x_input[:, 0]
        sigma = x_input[:, 1]
        # Complex valued of course....
        wave_number_real = epsilon * mu_0 * omega ** 2
        wave_number_imag = - sigma * omega
        # I Dont think we need this anymore...
        # self.wave_number_real = torch.nn.Parameter(torch.tensor(wave_number_real), requires_grad=False)
        # self.wave_number_imag = torch.nn.Parameter(torch.tensor(wave_number_imag), requires_grad=False)

        n_chan = pred.shape[1]
        loss = 0
        for i_chan in range(n_chan):
            pred_xx = self._pad(self.operator_xx(pred[:, i_chan:i_chan+1]))
            pred_yy = self._pad(self.operator_yy(pred[:, i_chan:i_chan+1]))

            target_xx = self._pad(self.operator_xx(target[:, i_chan:i_chan+1]))
            target_yy = self._pad(self.operator_yy(target[:, i_chan:i_chan+1]))

            # The channels are organized such that even == real valued.. odd == imag... valued
            # At least.. in most cases of course where we want to use this loss..
            if i_chan % 2 == 0:
                temp_wave_number = wave_number_real
                # temp_wave_number = self.wave_number_real
            else:
                temp_wave_number = wave_number_imag
                # temp_wave_number = self.wave_number_imag

            helmholtz_pred = pred_xx + pred_yy + self.dx ** 2 * temp_wave_number * pred[:, i_chan:i_chan+1]
            helmholtz_true = target_xx + target_yy + self.dx ** 2 * temp_wave_number * target[:, i_chan:i_chan+1]

            loss += (1/n_chan) * self.loss_obj(helmholtz_pred, helmholtz_true)
        return loss


class HelmholtzLoss(torch.nn.Module):
    def __init__(self, epsilon=50, mu=1e-4, omega=42, sigma=0.1, dx=1/256, kernel='6'):
        super().__init__()
        wave_number = epsilon * mu * omega ** 2 - 1j * sigma * omega
        # Had to do this because on the server we cant have the newest pytorch version
        self.wave_number_real = torch.nn.Parameter(torch.tensor(wave_number.real), requires_grad=False)
        self.wave_number_imag = torch.nn.Parameter(torch.tensor(wave_number.imag), requires_grad=False)

        # Second order derivatives
        # Taken from wikipedia..
        kernel_2 = np.array([1, -2, 1])
        kernel_4 = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])
        kernel_6 = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
        kernel_8 = np.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560])
        kernel_dict = {'2': [kernel_2, 0], '4': [kernel_4, 1], '6': [kernel_6, 2], '8': [kernel_8, 3]}

        kernel, self.pad = kernel_dict[kernel]
        self.dx = dx
        n_kernel = len(kernel)

        kernel_xx = np.tile(kernel, n_kernel).reshape(n_kernel, -1)
        kernel_xx = torch.nn.Parameter(torch.from_numpy(kernel_xx).view(1, 1, n_kernel, n_kernel).float(), requires_grad=False)
        filter_xx = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1, bias=False)
        filter_xx.weight.data = kernel_xx
        filter_xx.weight.requires_grad = False
        filter_xx.kernel_size = n_kernel

        kernel_yy = np.tile(kernel.T, n_kernel).reshape(n_kernel, -1)
        kernel_yy = torch.nn.Parameter(torch.from_numpy(kernel_yy).view(1, 1, n_kernel, n_kernel).float(), requires_grad=False)
        filter_yy = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1, bias=False)
        # Taking the transpose to transform it to a proper y-directional kernel
        filter_yy.weight.data = kernel_yy
        filter_yy.weight.requires_grad = False
        filter_yy.kernel_size = n_kernel

        self.operator_xx = filter_xx
        self.operator_yy = filter_yy

        self.loss_obj = torch.nn.MSELoss()

    def _pad(self, x):
        return F.pad(x, (self.pad, self.pad, self.pad, self.pad))

    def forward(self, pred, target):
        n_chan = pred.shape[1]
        loss = 0
        for i_chan in range(n_chan):
            pred_xx = self._pad(self.operator_xx(pred[:, i_chan:i_chan+1]))
            pred_yy = self._pad(self.operator_yy(pred[:, i_chan:i_chan+1]))

            target_xx = self._pad(self.operator_xx(target[:, i_chan:i_chan+1]))
            target_yy = self._pad(self.operator_yy(target[:, i_chan:i_chan+1]))

            # The channels are organized such that even == real valued.. odd == imag... valued
            # At least.. in most cases of course where we want to use this loss..
            if i_chan % 2 == 0:
                #temp_wave_number = self.wave_number.real
                temp_wave_number = self.wave_number_real
            else:
                #temp_wave_number = self.wave_number.imag
                temp_wave_number = self.wave_number_imag

            helmholtz_pred = pred_xx + pred_yy + self.dx ** 2 * temp_wave_number * pred[:, i_chan:i_chan+1]
            helmholtz_true = target_xx + target_yy + self.dx ** 2 * temp_wave_number * target[:, i_chan:i_chan+1]

            loss += (1/n_chan) * self.loss_obj(helmholtz_pred, helmholtz_true)
        return loss


class HelmholtzLoss1D(torch.nn.Module):
    def __init__(self, wave_number=(1.0, 1.0), dx=1.0, kernel='6'):
        super().__init__()
        self.wave_number = torch.nn.Parameter(torch.tensor(wave_number), requires_grad=False)
        # Second order derivatives
        kernel_2 = np.array([1, -2, 1])
        kernel_4 = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])
        kernel_6 = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
        kernel_8 = np.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560])
        kernel_dict = {'2': [kernel_2, 0], '4': [kernel_4, 1], '6': [kernel_6, 2], '8': [kernel_8, 3]}

        kernel, self.pad = kernel_dict[kernel]
        self.dx = dx
        n_kernel = len(kernel)

        kernel_xx = kernel
        kernel_xx = torch.nn.Parameter(torch.from_numpy(kernel_xx).view(1, 1, n_kernel).float(), requires_grad=False)
        filter_xx = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1, bias=False)
        filter_xx.weight.data = kernel_xx
        filter_xx.weight.requires_grad = False
        filter_xx.kernel_size = n_kernel

        self.operator_xx = filter_xx

        self.loss_obj = torch.nn.MSELoss()

    def _pad(self, x):
        return F.pad(x, (self.pad, self.pad))

    def forward(self, y_pred, y):
        # y is a tensor containing the boundary conditions
        # temp_y, should have no boundary conditions.. and just zero
        temp_y = torch.clone(y)
        temp_y[:, :, 0:10] = 0
        temp_y[:, :, -10:] = 0
        n_chan = y_pred.shape[1]
        loss = 0
        counter = 0
        for i_chan in range(n_chan):
            helmholtz_pred = self._pad(self.operator_xx(y_pred[:, i_chan:i_chan+1])) + \
                             self.dx ** 2 * self.wave_number[counter] * y_pred[:, i_chan:i_chan+1]
            # Hier is y dus het resultaat NA de differentiaal operator
            loss += (1/n_chan) * self.loss_obj(helmholtz_pred, temp_y[:, i_chan:i_chan+1]) + \
                    0.1 * self.loss_obj(y_pred[:, :, 0:10], y[:, :, 0:10]) + \
                    0.1 * self.loss_obj(y_pred[:, :, -10:], y[:, :, -10:])
            counter += 1
            # (The 0.1 above is a loss for the boundary conditions...)
        return loss


class LaPlaceLoss1D(torch.nn.Module):
    def __init__(self, dx=1.0, kernel='6'):
        super().__init__()
        # Second order derivatives
        kernel_2 = np.array([1, -2, 1])
        kernel_4 = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])
        kernel_6 = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
        kernel_8 = np.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560])
        kernel_dict = {'2': [kernel_2, 0], '4': [kernel_4, 1], '6': [kernel_6, 2], '8': [kernel_8, 3]}

        kernel, self.pad = kernel_dict[kernel]
        self.dx = dx
        n_kernel = len(kernel)

        kernel_xx = kernel
        kernel_xx = torch.nn.Parameter(torch.from_numpy(kernel_xx).view(1, 1, n_kernel).float(), requires_grad=False)
        filter_xx = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1, bias=False)
        filter_xx.weight.data = kernel_xx
        filter_xx.weight.requires_grad = False
        filter_xx.kernel_size = n_kernel

        self.operator_xx = filter_xx

        self.loss_obj = torch.nn.MSELoss()

    def _pad(self, x):
        return F.pad(x, (self.pad, self.pad))

    def forward(self, y_pred, y):
        # y is a tensor containing the boundary conditions
        # temp_y, should have no boundary conditions.. and just zero
        temp_y = torch.clone(y)
        temp_y[:, :, 0:10] = 0
        temp_y[:, :, -10:] = 0
        n_chan = y_pred.shape[1]
        loss = 0
        counter = 0
        for i_chan in range(n_chan):
            laplace_pred = self._pad(self.operator_xx(y_pred[:, i_chan:i_chan+1]))
            # Hier is y dus het resultaat NA de differentiaal operator
            loss += (1/n_chan) * self.loss_obj(laplace_pred, temp_y[:, i_chan:i_chan+1]) + \
                           0.1 * self.loss_obj(y_pred[:, :, 0:10], y[:, :, 0:10]) + \
                           0.1 * self.loss_obj(y_pred[:, :, -10:], y[:, :, -10:])
            counter += 1
            # (The 0.1 above is a loss for the boundary conditions...)
        return loss


if __name__ == "__main__":

    """
    Test if we can get np gradient and torch.diff (with convolution) to be the same
    """

    def get_first_derivative(x, dx, axis=0):
        return np.gradient(x, axis=axis) * (1/dx)

    def get_second_derivative(x, dx, axis=0):
        return np.gradient(np.gradient(x, axis=axis) * (1/dx), axis=axis) * (1/dx)

    """
    Test how Helmholtz Loss behaves
    """

    x_range = np.arange(-3, 3, 0.005)
    X, Y = np.meshgrid(x_range, x_range)
    dx = np.diff(x_range)[0]
    Z = np.cos(X) + np.sin(Y)
    Z = np.random.random(X.shape)
    Z_target = np.zeros(X.shape)
    Z_x = -np.sin(X)
    Z_xx = -np.cos(X)
    Z_y = np.cos(Y)
    Z_yy = -np.sin(Y)
    plt.imshow(Z)
    Z_tens = torch.as_tensor(Z[None, None]).float()
    Z_tens_target = torch.as_tensor(Z_target[None, None]).float()

    helm_loss = HelmholtzLoss(dx=dx)
    with torch.no_grad():
        res = helm_loss(Z_tens, Z_tens_target)

    # Uitkleden van loss voor stap voor stap berenkening
    i_chan = 0

    helmholtz_pred = helm_loss._pad(helm_loss.operator_xx(Z_tens[:, i_chan:i_chan+1])) + helm_loss.dx ** 2 * helm_loss.wave_number * Z_tens[:, i_chan:i_chan+1]

    plt.imshow(helmholtz_pred.detach().numpy()[0][0])
    helm_loss.loss_obj(helmholtz_pred, torch.from_numpy(np.zeros((1, 1, 1200, 1200))))


    """
    1D HelmHoltz Loss
    """
    import numpy as np
    x_range = np.arange(-3, 3, 0.005)
    dx = np.diff(x_range)[0]
    Z = np.random.random(x_range.shape)
    Z_target = np.zeros(x_range.shape)

    Z_tens = torch.as_tensor(Z[None, None]).float()
    Z_tens = torch.cat([Z_tens, Z_tens], axis=1)
    Z_tens_target = torch.as_tensor(Z_target[None, None]).float()
    Z_tens_target = torch.cat([Z_tens_target, Z_tens_target], axis=1)
    helm_obj = HelmholtzLoss1D()
    helm_obj(Z_tens, Z_tens_target)
    #
    helm_obj.operator_xx(Z_tens).shape
    z = helm_obj._pad(helm_obj.operator_xx(Z_tens[:, i_chan:i_chan+1])) + \
    helm_obj.dx ** 2 * helm_obj.wave_number * Z_tens[:, i_chan:i_chan+1]
    z.shape
    helm_obj.loss_obj(z, Z_tens_target)

    """Laplace loss"""
    laplace_loss = LaPlaceLoss1D()
    x_range = np.linspace(-2, 2, 50)
    y_range = x_range ** 2
    y_tensor = torch.from_numpy(y_range).float()
    laplace_pred = laplace_loss._pad(laplace_loss.operator_xx(y_tensor[None, None]))
    import matplotlib.pyplot as plt
    plt.plot(laplace_pred.numpy()[0][0])
