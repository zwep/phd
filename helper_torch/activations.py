# encoding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc

"""

"""
# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.parameter


class GaussianRBF(nn.Module):
    def __init__(self, epsilon=1):
        super().__init__()
        self.epsilon = torch.nn.parameter.Parameter(torch.tensor(epsilon), requires_grad=True)

    def forward(self, x):
        return torch.exp(-(self.epsilon * x)**2)


class SinExponential(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        # initialize alpha
        self.alpha = torch.nn.parameter.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, x):
        return torch.sin(x) * torch.exp(-(self.alpha * x) ** 2)


class SoftExponential(nn.Module):
    def __init__(self, alpha=0.0):
        super().__init__()
        # initialize alpha
        self.alpha = torch.nn.parameter.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        if self.alpha == 0.0:
            return x
        if self.alpha < 0.0:
            return - torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha
        if self.alpha > 0.0:
            return (torch.exp(self.alpha * x) - 1) / self.alpha + self.alpha


class ArctanTanExp(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0, delta=0.0):
        super().__init__()
        # initialize alpha
        self.alpha = torch.nn.parameter.Parameter(torch.tensor(alpha), requires_grad=True)
        self.beta = torch.nn.parameter.Parameter(torch.tensor(beta), requires_grad=True)
        self.gamma = torch.nn.parameter.Parameter(torch.tensor(gamma), requires_grad=True)
        self.delta = torch.nn.parameter.Parameter(torch.tensor(delta), requires_grad=True)

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        return 2 * torch.atan(self.alpha * torch.tan(self.beta * x)) * torch.exp(-0.5*((x - self.gamma)/self.delta) ** 2)


class ArctanTan(nn.Module):
    def __init__(self, scale_factor=2/np.pi, alpha=1.0, beta=1.0, gamma=0.0, delta=0.0):
        # Scale factor is 2/pi, because au
        super().__init__()
        # initialize alpha
        self.scale_factor = scale_factor
        self.alpha = torch.nn.parameter.Parameter(torch.tensor(alpha), requires_grad=True)
        self.beta = torch.nn.parameter.Parameter(torch.tensor(beta), requires_grad=True)
        self.gamma = torch.nn.parameter.Parameter(torch.tensor(gamma), requires_grad=True)
        self.delta = torch.nn.parameter.Parameter(torch.tensor(delta), requires_grad=True)

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        return self.scale_factor * torch.atan(self.alpha * torch.tan(self.beta * x + self.gamma) + self.delta)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SoftPluss(nn.Module):
    def __init__(self, alpha=3.14, beta=1.0):
        super().__init__()
        self.alpha = torch.nn.parameter.Parameter(torch.tensor(alpha), requires_grad=True)
        self.beta = torch.nn.parameter.Parameter(torch.tensor(beta), requires_grad=True)

    def forward(self, x):
        return self.beta * torch.tanh(self.alpha * x)


class ScaledTanh(nn.Module):
    def __init__(self, alpha=3.14, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        return self.alpha * torch.tanh(x)


class LearnTanh(nn.Module):
    def __init__(self, alpha=3.14, beta=1.0):
        super().__init__()
        self.alpha = torch.nn.parameter.Parameter(torch.tensor(alpha), requires_grad=True)
        self.beta = torch.nn.parameter.Parameter(torch.tensor(beta), requires_grad=True)

    def forward(self, x):
        return self.beta * torch.tanh(self.alpha * x)


class LearnSin(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = torch.nn.parameter.Parameter(torch.tensor(alpha), requires_grad=True)
        self.beta = torch.nn.parameter.Parameter(torch.tensor(beta), requires_grad=True)

    def forward(self, x):
        return self.beta * torch.sin(self.alpha * x)


class ArcSinh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(x+(x**2+1)**0.5)


class GatedConditionalPixel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_f, h_f, x_g, h_g):
        return torch.tanh(x_f + h_f) * torch.sigmoid(x_g + h_g)


def get_atantan(x, alpha, beta, gamma, delta):
    return 2 * np.arctan(alpha * np.tan(beta * x + gamma) + delta)


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm

    derp = soft_exponential((1,), 0.1)
    derp(5)
    derp2 = ArctanTan(0.5, 0.5)
    derp2(1)

    x_range = np.arange(-10, 10, 0.1)
    cmap = matplotlib.cm.get_cmap('Reds')
    cmap = matplotlib.cm.get_cmap('Greens')
    cmap = matplotlib.cm.get_cmap('Blues')
    i_beta = 0.95
    for i_c, i_alpha in enumerate(np.arange(-10, 0, 0.5)):
        y = get_atantan(x_range, i_alpha, i_beta)
        plt.plot(x_range, y, color=cmap(i_c * 10 + 50))

    i_alpha = 1
    cmap = matplotlib.cm.get_cmap('Reds')
    # Value clipping tussen -1 en 1 ofzo...
    for i_c, i_beta in enumerate(np.arange(-2, 2, 0.5)):
        y = get_atantan(x_range, i_alpha, i_beta)
        plt.plot(x_range, y, color=cmap(i_c * 20 + 50))
        # plt.plot(x_range, y+np.pi)

    x = np.arange(-20, 20, 0.1)
    y = 2*np.arctan(np.tan(x)) * np.exp(-0.1 * x ** 2/10)
    # y = np.arctan(np.tan(x)) * np.arctanh(0.005 * x) ** 2
    plt.plot(x, y)
