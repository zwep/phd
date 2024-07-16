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
import torch.nn
import helper_torch.layers as hlayers


class DeepDenseNet(torch.nn.Module):
    def __init__(self, n_pool_layers):
        # Expects input of ...
        super().__init__()
        self.layer_list = torch.nn.ModuleList()
        self.dense_layer = torch.nn.Sequential(torch.nn.Linear(in_features=8, out_features=16),
                                               torch.nn.Linear(in_features=16, out_features=8))
        self.dense_final = torch.nn.Linear(in_features=(n_pool_layers-1)*16 + 48, out_features=16)
        self.to_last = hlayers.SwapAxes2D(to_channel_last=True)
        self.to_first = hlayers.SwapAxes2D(to_channel_last=False)
        for i in range(2*n_pool_layers+1):
            temp_sequence = torch.nn.Sequential(torch.nn.Linear(in_features=2, out_features=4),
                                                torch.nn.Linear(in_features=4, out_features=4),
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(in_features=4, out_features=2),
                                                hlayers.SwapAxes2D(to_channel_last=False),
                                                torch.nn.Dropout2d(),
                                                hlayers.SwapAxes2D(to_channel_last=True))
            temp_sequence.weights = torch.nn.Parameter(torch.as_tensor(np.eye(2)))
            self.layer_list.append(temp_sequence)

    def forward(self, x):
        x = self.to_last(x)
        x_list = x.split(2, dim=-1)
        n = len(x_list)
        stack = []
        for i, layer in enumerate(self.layer_list):
            if i % 2 == 0:
                x_stacked = torch.stack(x_list, dim=-1)
                x_stacked = self.dense_layer(x_stacked)
                x_list = [x[:, :, :, :, 0] for x in torch.split(x_stacked, 1, dim=-1)]
                stack.append(torch.cat(x_list, dim=-1))
            else:
                x_list = [layer(x) for x in x_list]

        stack_cat = torch.cat(stack, dim=-1)
        x_cat = torch.cat(x_list, dim=-1)
        x_final = torch.cat([stack_cat, x_cat], dim=-1)
        x_final = self.dense_final(x_final)
        return self.to_first(x_final)


if __name__ == "__main__":
    import numpy as np
    import torch
    import torch.nn
    import helper.plot_fun as hplotf
    import helper.array_transf as harray
    import helper.plot_fun as hplotf
    import helper.plot_class as hplotc
    import matplotlib.pyplot as plt

    import data_generator.Rx2Tx as gen_rx2tx

    import helper_torch.transforms as htransform
    import os
    import numpy as np
    import scipy.stats
    import importlib

    dir_data = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels'
    dir_data_svd = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels_svd'

    # Test SURVEY2B1_all - line by line view
    importlib.reload(gen_rx2tx)

    dg_gen_rx2tx = gen_rx2tx.DataSetSurvey2B1_all(input_shape=(16, 512, 256), ddata=dir_data,
                                                  input_is_output=False, number_of_examples=2,
                                                  transform_type='complex', complex_type='cartesian', shuffle=False,
                                                  relative_phase=False, masked=True,
                                                  fourier_transform=False)
    a, b = dg_gen_rx2tx.__getitem__(0)
    a = a[np.newaxis]
    b = b[np.newaxis]
    res = DeepDenseNet(n_pool_layers=1)
    res_A = res(a)
    hplotf.plot_3d_list(a.numpy())
    hplotf.plot_3d_list(res_A.detach().numpy())