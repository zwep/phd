import helper_torch.loss as htloss
import torch

import helper.array_transf as harray
import helper.plot_class as hplotc
import helper.dummy_data as hdummy

import numpy as np
import os

x_ellipse, _ = hdummy.get_elipse(100, 100)
x_tens = torch.from_numpy(x_ellipse[None])
# x_transf = torch_transf(x_tens)

hplotc.ListPlot([x_transf, x_ellipse])

loss_obj = htloss.FocalLoss(alpha=0.25)
loss_obj(x_transf, x_tens)

input = x_transf[None]
A = torch.empty(5, 3, 5, 5, dtype=torch.long).random_(2)
# See how easily we can go from binary to N-class target

torch.nonzero(A==1)


import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs.squeeze(),  targets.float())
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss

