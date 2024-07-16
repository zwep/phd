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


class GroupedXNetDown(torch.nn.Module):
    """
     Down scale part of a UNET
     """

    def __init__(self, in_chans=2, out_chans=2, chans=2, num_pool_layers=3, drop_prob=0.1, **kwargs):
        super().__init__()

        self.debug = kwargs.get('debug')
        convblock_activation = kwargs.get('convblock_activation', 'relu')
        groups = kwargs.get('conv_groups', 2)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = torch.nn.ModuleList(
            [ConvBlock2D(in_chans=in_chans, out_chans=chans, drop_prob=drop_prob,
                         convblock_activation=convblock_activation, groups=groups,
                         debug=self.debug)])
        ch = chans
        for i in range(num_pool_layers - 1):
            print(f'GROUPED-DOWN:  chan {ch}  groups {groups}')
            self.down_sample_layers += [ConvBlock2D(ch, ch * 2, drop_prob=drop_prob, groups=groups,
                                                    convblock_activation=convblock_activation)]
            ch *= 2

        self.conv = ConvBlock2D(ch, ch, drop_prob=drop_prob, groups=groups,
                                convblock_activation=convblock_activation)
        self.pool = torch.nn.AvgPool2d(kernel_size=2)

    def forward(self, input):
        # Stack is used to to use as skip connection in the UNET
        stack = []
        output = input
        debug_text = []

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            debug_text.append('layer i {} - {}'.format(i, len(self.down_sample_layers)))
            output = layer(output)
            stack.append(output)
            output = self.pool(output)

        output = self.conv(output)

        if self.debug:
            print('GROUPED-DOWN: ')
            print('\n'.join(debug_text))

        return output, stack


class GroupedXNetFeature(torch.nn.Module):
    """
    Used to combine the output of multiple Down scaled Unet models
    """
    def __init__(self, n_concat, n_hidden, **kwargs):
        # Could change it here to a Sequential thing instead of a Module
        super().__init__()
        self.debug = kwargs.get('debug')
        linear_layer_1 = torch.nn.Linear(in_features=n_concat, out_features=n_hidden)
        linear_layer_2 = torch.nn.Linear(in_features=n_hidden, out_features=n_concat)

        if n_concat == n_hidden:
            print('GROUPED-FTR: using eye matrix as init')
            linear_layer_1.weight = torch.nn.Parameter(torch.as_tensor(np.eye(n_hidden)), requires_grad=True)
            linear_layer_2.weight = torch.nn.Parameter(torch.as_tensor(np.eye(n_hidden)), requires_grad=True)
        self.layer_list = torch.nn.ModuleList([linear_layer_1, linear_layer_2])

    def forward(self, x):
        for i_layer in self.layer_list:
            x = i_layer(x)
            if self.debug:
                print('GROUPED-FTR: ', x.shape)
        return x


class GroupedXNetUp(torch.nn.Module):
    """
      Upscaled version of the UNET
      """

    def __init__(self, in_chans=1, out_chans=1, num_pool_layers=3, drop_prob=0.1, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug')
        self.groups = kwargs.get('groups', 2)
        self.output_activation = kwargs.get('output_activation', 'identity')
        self.convblock_activation = kwargs.get('convblock_activation', 'relu')
        ch = in_chans // 2  # I have no idea why this guy is here... but w/e
        self.up_sample_layers = torch.nn.ModuleList()

        for i in range(num_pool_layers - 1):
            print(f'GROUPED UP: amount of channels {ch} groups {self.groups}')
            self.up_sample_layers += [ConvBlock2D(ch * 2, ch // 2, drop_prob=drop_prob, groups=self.groups,
                                                  convblock_activation=self.convblock_activation,
                                                  debug=self.debug)]
            ch //= 2
        self.up_sample_layers += [ConvBlock2D(ch * 2, ch, drop_prob=drop_prob,
                                              convblock_activation=self.convblock_activation,
                                              groups=self.groups)]
        print(f'GROUPED UP: amount of channels {ch} groups {self.groups}')
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(ch, ch // 2, kernel_size=1),
                                         torch.nn.Conv2d(ch // 2, out_chans, kernel_size=1),
                                         torch.nn.Conv2d(out_chans, out_chans, kernel_size=1))
        self.linear_layer = torch.nn.Linear(in_features=out_chans, out_features=out_chans)
        weight_tens = torch.as_tensor(np.diag([(-1) ** i for i in range(out_chans)])).float()
        self.linear_layer.weight = torch.nn.Parameter(weight_tens, requires_grad=True)
        self.output_actv_layer = activation_selector(self.output_activation)

    def forward(self, input, stack):
        # Apply up-sampling layers
        stack = list(stack)
        output = input
        debug_text = []
        for i, layer in enumerate(self.up_sample_layers):
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)
            if self.debug:
                debug_text.append('layer i {} - {}'.format(i, len(self.up_sample_layers)))
                debug_text.append('\t down sample layer {}'.format(downsample_layer.shape))
                debug_text.append('\t interp layer size {}'.format(layer_size))
                debug_text.append('\t output size {}'.format(output.shape))
                debug_text.append('\t layer {}'.format(layer))

        if self.debug:
            print('\n'.join(debug_text))
        output = self.conv2(output)

        if self.output_activation is not None:
            output = self.output_actv_layer(output)

        return output


class GroupedXNet(torch.nn.Module):
    """
    Compsite model of the ones defined above...
    """
    def __init__(self, start_chan, out_chans, n_pool_layers, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug')
        self.device = kwargs.get('device')
        output_activation = kwargs.get('output_activation', 'identity')
        convblock_activation = kwargs.get('convblock_activation', 'relu')
        down_block = kwargs.get('down_block', 'ConvBlock')
        up_block = kwargs.get('up_block', 'ConvBlock')

        n_concat = int((start_chan / 2) * 2 ** n_pool_layers)

        self.mod_down = GroupedXNetDown(in_chans=2, chans=start_chan, num_pool_layers=n_pool_layers,
                                        convblock_activation=convblock_activation, down_block=down_block,
                                        debug=self.debug)
        self.mod_ftr = GroupedXNetFeature(n_concat=8, n_hidden=8, debug=self.debug)
        self.mod_up = GroupedXNetUp(in_chans=2*n_concat, out_chans=out_chans, num_pool_layers=n_pool_layers,
                                    output_activation=output_activation,
                                    convblock_activation=convblock_activation,
                                    up_block=up_block, debug=self.debug)
        self.dense_seq = [torch.nn.Sequential(hlayers.SwapAxes2D(to_channel_last=True),
                                              torch.nn.Linear(2, 2),
                                              hlayers.SwapAxes2D(to_channel_last=False)) for _ in range(8)]
        for i in range(8):
            self.dense_seq[i][1].weight = torch.nn.Parameter(torch.as_tensor(np.eye(2)), requires_grad=True)
            self.dense_seq[i][1].float()

        [x.to(self.device) for x in self.dense_seq]

    def forward(self, x):
        input_tensor_list = torch.split(x, 2, dim=1)
        input_tensor_list = [self.dense_seq[i](x) for i, x in enumerate(input_tensor_list)]  # Model...

        output_stack_mod_down = [self.mod_down(x) for x in input_tensor_list]  # Model... down...
        result_down, stack_mod_down = zip(*output_stack_mod_down)
        cat_result_down_perm = torch.stack(result_down, dim=-1)
        result_mid_perm = self.mod_ftr(cat_result_down_perm)
        result_mid_split = result_mid_perm.split(1, dim=-1)
        result_mid_split = [x[:, :, :, :, 0] for x in result_mid_split]  # take the last dimension as result of split

        result_up = [self.mod_up(x, stack=y) for x, y in zip(result_mid_split, stack_mod_down)]  # Model...
        output = torch.cat(result_up, dim=1)

        return output


if __name__ == "__main__":
    import torch
    import torch.utils.data
    import numpy as np
    import data_generator.Rx2Tx as gen_rx2tx

    model_obj = GroupedXNet(debug=True, start_chan=2, out_chans=2, n_pool_layers=3, groups=2).float()

    dir_data = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels'

    dg_gen_rx2tx_svd = gen_rx2tx.DataSetSurvey2B1_all_svd(input_shape=(16, 512, 256), ddata=dir_data + '_svd',
                                                          input_is_output=False, number_of_examples=1,
                                                          transform_type='complex', complex_type='cartesian',
                                                          shuffle=False)

    a, b = dg_gen_rx2tx_svd.__getitem__(0)
    res_a = model_obj.forward(a[np.newaxis])
    hplotf.plot_3d_list(a[np.newaxis])
    hplotf.plot_3d_list(b[np.newaxis])
    hplotf.plot_3d_list(res_a.detach().numpy(), cbar=True, vmin=(0.4, 0.6))

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
