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

import torch
import torch.nn
import torch.nn.functional as F
import model.Blocks as Blocks


class GeneratorConditionalGan(torch.nn.Module):
    raise DeprecationWarning
    # Been trying it... not working.
    def __init__(self, in_chan=2, out_chan=2, start_ch=2, n_pool_layers=5, n_classes=8, groups=2, embedding_shape=(16, 8), **kwargs):
        # Embedding size is chosen in such a way that 16 * 8 = 128
        super().__init__()
        self.debug = kwargs.get('debug')
        self.device = kwargs.get('device', 'cpu')
        self.embedding_shape = embedding_shape
        embedding_size = int(np.prod(self.embedding_shape))
        self.embedding = torch.nn.Embedding(num_embeddings=n_classes, embedding_dim=embedding_size)
        self.index_tensor = torch.Tensor(list(range(n_classes))).long()  # Define it here, hope that it gets transfered to the GPU.
        self.index_tensor.requires_grad = False
        self.groups = groups

        down_sample = []

        first_conv = torch.nn.Conv2d(in_channels=in_chan, out_channels=start_ch, kernel_size=1, groups=1)
        down_sample.append(first_conv)

        ch = start_ch
        for i_layer in range(n_pool_layers):
            temp_down = Blocks.ConvBlock2D(in_chans=ch, out_chans=2 * ch, groups=self.groups)
            temp_sample = torch.nn.Conv2d(in_channels=2 * ch, out_channels=2 * ch, kernel_size=2, stride=2, groups=1)
            down_sample.append(temp_down)
            down_sample.append(temp_sample)
            ch = 2 * ch

        self.down_sample = torch.nn.Sequential(*down_sample)

        up_sample = []
        for i_layer in range(n_pool_layers):
            if i_layer == 0:
                temp_up = torch.nn.Conv2d(in_channels=ch+2, out_channels=ch, kernel_size=1, stride=1, groups=1)
            else:
                temp_up = torch.nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1, stride=1, groups=1)
            if self.debug:
                print(f'GEN AC GAN up channels {ch}, {ch//2} \t {self.groups}')
            temp_conv = Blocks.ConvBlock2D(in_chans=ch, out_chans=max(self.groups, ch // 2), groups=self.groups)

            temp_seq = torch.nn.Sequential(temp_up, temp_conv)
            up_sample.append(temp_seq)
            ch = ch // 2

        self.up_sample = torch.nn.ModuleList(up_sample)

        self.last_conv = torch.nn.Conv2d(in_channels=ch, out_channels=out_chan, kernel_size=1, groups=1)

        # self.n_concat = n_classes   # * start_ch * 2 ** n_pool_layers
        self.n_classes = n_classes
        self.linear_layer = torch.nn.Linear(in_features=self.n_classes, out_features=self.n_classes)

    def up_layer(self, x, fun):
        x_shape = x.shape
        x_new_shape = (2 * x_shape[-2], 2 * x_shape[-1])
        x = F.interpolate(x, x_new_shape)
        x = fun(x)
        return x

    def forward(self, x):
        # Hopsa, gewoon hier in gooien.
        z = torch.as_tensor(np.random.normal(0, 1, size=(1, self.groups) + self.embedding_shape)).float()
        z = z.to(self.device)

        # X is the input of the coils.. (?, 8, X, Y)
        # Z is the input of the noise.. should have shape (16, 8)

        x_split = x.split(split_size=2, dim=1)
        x_down = [self.down_sample(xi) for xi in x_split]
        if self.debug:
            print('x down', x_down[0].shape)
        down_stacked = torch.stack(x_down, dim=-1)
        if self.debug:
            print('down stacked', down_stacked.shape)
            print('linear layer', self.linear_layer)
        stacked_linear = self.linear_layer(down_stacked)
        if self.debug:
            print('linear layer', stacked_linear.shape)
        linear_split = stacked_linear.split(split_size=1, dim=-1)
        linear_split = [ix[:, :, :, :, 0] for ix in linear_split]
        # Add noise over here.. and coil embedding.. per coil.
        for i_ind in range(len(linear_split)):
            temp_noise = z[:, i_ind:i_ind+1]
            if self.debug:
                print(f'noise shape {temp_noise.shape}')
            temp_coil_embed = self.embedding(self.index_tensor[i_ind]).reshape((1, 1) + self.embedding_shape)
            if self.debug:
                print(f'embedding shape {temp_coil_embed.shape}')
                print(f'linear split shape {linear_split[i_ind].shape}')
            linear_split[i_ind] = torch.cat([linear_split[i_ind], temp_noise, temp_coil_embed], dim=1)

        split_up = []
        for temp in linear_split:
            for i_up in self.up_sample:
                temp = self.up_layer(temp, i_up)
                if self.debug:
                    print(f'temp shape {temp.shape}')
                    print(f'layer {i_up}')

            split_up.append(temp)

        split_up = [self.last_conv(x) for x in split_up]
        split_up = torch.cat(split_up, dim=1)

        return split_up


if __name__ == "__main__":
    # Load data
    import os
    import helper.plot_fun as hplotf
    import numpy as np
    import helper.array_transf as harray
    import data_generator.Rx2Tx as dg_rxtx

    dg = dg_rxtx.DataSetSurvey2B1_flavio(ddata='/home/bugger/Documents/data/simulation/flavio_npy',
                                         input_shape=(16, 256, 256),
                                         masked=True)

    dg = dg_rxtx.DataSetSurvey2B1_all(ddata='/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels',
                                      input_shape=(16, 512, 256),
                                      masked=True)
    a_tens, b, mask = dg.__getitem__(0)

    # Get model
    n_classes = 2
    chan, ny, nx = (2, 256, 256)# a_tens.shape
    n_pool_layer = 3
    embedding_shape = (int(ny / 2**n_pool_layer), int(nx / 2**n_pool_layer))
    mod_obj = GeneratorConditionalGan(in_chan=2, out_chan=2, start_ch=4, n_pool_layers=n_pool_layer, n_classes=n_classes, groups=1,
                                      embedding_shape=embedding_shape, debug=True)
    # z_noise = torch.as_tensor(np.random.rand(*((1, 8) + embedding_shape))).float()
    mod_obj.embedding_shape
    res = mod_obj(a_tens[np.newaxis, 0:2])
    hplotf.plot_3d_list(res.detach().numpy())
    hplotf.plot_3d_list(a_tens[:, 0:4].detach().numpy())
    len(res)
    res[0].shape