# encoding: utf-8

# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
from helper_torch.misc import activation_selector, block_selector

import torch
import torch.nn
import torch.nn.functional as F
import torch.utils
import helper_torch.activations as hactv
import helper.misc as hmisc
from model.Blocks import ConvBlock2D, GroupedConvBlock
import helper_torch.layers as hlayers


class GatedXNetDown(torch.nn.Module):
    """
    Down scale part of a UNET
    """

    def __init__(self, in_chans=1, out_chans=1, chans=2, num_pool_layers=3, drop_prob=0.1, **kwargs):
        super().__init__()

        self.debug = kwargs.get('debug')
        self.pool = torch.nn.AvgPool2d(kernel_size=2)
        self.grouped_block1 = GroupedConvBlock(in_chans=4, out_chans=8, drop_prob=0.1)
        self.grouped_block2 = GroupedConvBlock(in_chans=8, out_chans=16, drop_prob=0.1)

    def forward(self, x):
        x = self.grouped_block1(x)
        if self.debug:
            print('GATED-DOWN: ', x.shape)
        x = self.pool(x)
        if self.debug:
            print('GATED-DOWN: ', x.shape)
        x = self.grouped_block2(x)
        if self.debug:
            print('GATED-DOWN: ', x.shape)
        x = self.pool(x)
        if self.debug:
            print('GATED-DOWN: ', x.shape)
        return x


class GatedXNetFeature(torch.nn.Module):
    """
    Used to combine the output of multiple Down scaled Unet models
    """
    def __init__(self, n_concat, n_hidden, **kwargs):
        # Could change it here to a Sequential thing instead of a Module
        super().__init__()
        self.debug = kwargs.get('debug')
        self.layer_list = torch.nn.ModuleList([torch.nn.Linear(in_features=n_concat, out_features=n_hidden),
                                               torch.nn.Linear(in_features=n_hidden, out_features=n_concat)])

    def forward(self, x):
        for i_layer in self.layer_list:
            x = i_layer(x)
            if self.debug:
                print('GATED-FTR: ', x.shape)
        return x


class GatedXNetUp(torch.nn.Module):
    """
    Upscaled version of the UNET
    """

    def __init__(self, in_chans=1, out_chans=1, num_pool_layers=3, drop_prob=0.1, output_activation=None,
                 convblock_activation=None, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug')
        self.grouped_block1 = GroupedConvBlock(in_chans=16, out_chans=8, drop_prob=0.1)
        self.grouped_block2 = GroupedConvBlock(in_chans=8, out_chans=4, drop_prob=0.1)

    def forward(self, x):
        x = self.grouped_block1(x)
        if self.debug:
            print('GATED-UP: ', x.shape)
        batch, chan, xdim, ydim = x.shape
        x = F.interpolate(x, size=(2*xdim, 2*ydim), mode='bilinear', align_corners=False)
        if self.debug:
            print('GATED-UP: ', x.shape)
        x = self.grouped_block2(x)
        if self.debug:
            print('GATED-UP: ', x.shape)
        batch, chan, xdim, ydim = x.shape
        x = F.interpolate(x, size=(2 * xdim, 2 * ydim), mode='bilinear', align_corners=False)
        if self.debug:
            print('GATED-UP: ', x.shape)

        return x


class GatedXNet(torch.nn.Module):
    """
    Compsite model of the ones defined above...
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug')
        self.mod_down = GatedXNetDown(debug=self.debug)
        self.mod_ftr = GatedXNetFeature(n_concat=8, n_hidden=8, debug=self.debug)
        self.mod_up = GatedXNetUp(debug=self.debug)
        self.dense_seq = [torch.nn.Sequential(hlayers.SwapAxes2D(to_channel_last=True),
                                              torch.nn.Linear(2, 2),
                                              hlayers.SwapAxes2D(to_channel_last=False)) for _ in range(8)]
        self.split_pos_neg = hlayers.SplitPosNegLayer()
        self.conv = torch.nn.Conv2d(4*8, 16, kernel_size=1)

    def forward(self, x):
        input_tensor_list = torch.split(x, 2, dim=1)
        if self.debug:
            print('GATED_XNET: ', len(input_tensor_list), input_tensor_list[0].shape)
        input_tensor_list = [self.dense_seq[i](x) for i, x in enumerate(input_tensor_list)]  # Model...
        if self.debug:
            print('GATED_XNET: ', len(input_tensor_list), input_tensor_list[0].shape)
        input_tensor_list = [self.split_pos_neg(x) for x in input_tensor_list]
        if self.debug:
            print('GATED_XNET: ', len(input_tensor_list), input_tensor_list[0].shape)

        output_stack_mod_down = [self.mod_down(x) for x in input_tensor_list]  # Model... down...
        output_mod_down = torch.stack(output_stack_mod_down, dim=-1)
        ftr_out = self.mod_ftr(output_mod_down)
        split_ftr = ftr_out.split(1, dim=-1)
        res = [self.mod_up(x[:, :, :, :, 0]) for x in split_ftr]
        output = torch.cat(res, dim=1)

        return self.conv(output)


if __name__ == "__main__":
    import torch
    import torch.utils.data
    import numpy as np
    import data_generator.Rx2Tx as gen_rx2tx

    A = torch.as_tensor(np.random.normal(0, 1, size=(1, 16, 32, 32)))

    mod = GatedXNet(debug=False)
    dir_data = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels'

    dg_gen_rx2tx_svd = gen_rx2tx.DataSetSurvey2B1_all_svd(input_shape=(16, 512, 256), ddata=dir_data + '_svd',
                                                          input_is_output=False, number_of_examples=1,
                                                          transform_type='complex', complex_type='cartesian',
                                                          shuffle=False)
    dg_loader = torch.utils.data.DataLoader(dg_gen_rx2tx_svd, batch_size=2, shuffle=True, num_workers=0)

    import torch.optim as toptimizer
    mod.train()
    optim = toptimizer.Adam(lr=0.01, params=mod.parameters())
    loss_obj = torch.nn.L1Loss()
    loss_curve = []
    n_epoch = 100
    for epoch in range(n_epoch):
        for X, y in dg_loader:
            predy = mod(X)
            loss = loss_obj(predy, y)
            loss_curve.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()

        print(epoch, np.mean(loss_curve))

    hplotf.plot_3d_list(predy.detach().numpy())

