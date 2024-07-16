# encoding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn

import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
from model.XNet import XNet

"""

"""


class XXNet(torch.nn.Module):
    """
    Model architecture for trying out combinations for imag values..
    """
    def __init__(self, n_pool_layers=3, start_chan=4, n_hidden=512, **kwargs):
        super().__init__()
        n_concat = int((start_chan / 2) * 2 ** n_pool_layers * 8)
        self.xnet_real_real = XNet(n_pool_layers=n_pool_layers, start_chan=start_chan, n_hidden=n_hidden, **kwargs)
        self.xnet_imag_real = XNet(n_pool_layers=n_pool_layers, start_chan=start_chan, n_hidden=n_hidden, **kwargs)
        self.concat_layer = torch.nn.Linear(in_features=2, out_features=1)
        self.n_concat = n_concat

    def forward(self, x):
        input_real_imag = torch.split(x, 8, dim=1)

        output_1 = self.xnet_real_real(input_real_imag[0])
        output_2 = self.xnet_imag_real(input_real_imag[1])
        output_1 = output_1.unsqueeze(-1)
        output_2 = output_2.unsqueeze(-1)
        output = torch.cat([output_1, output_2], dim=-1)
        output = self.concat_layer(output)[:, :, :, :, 0]  # Deselecting the last tying

        return output


if __name__ == "__main__":

    # # # Test XX Net
    import data_generator.Rx2Tx as dg_rxtx
    from torch.utils.data import DataLoader
    import model.XXNet
    import importlib

    importlib.reload(model.XNet)


    ddata = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels'
    DG = dg_rxtx.DataSetSurvey2B1_all(ddata, input_shape=(16, 512, 256), transform_type='complex')
    n_files = len(DG)
    batch_size = 5
    batch_size = hmisc.correct_batch_size(batch_size, n_files)
    print('batch size', batch_size)
    data_loader = torch.utils.data.DataLoader(DG, batch_size=batch_size)

    ddata = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels'
    DG = dg_rxtx.DataSetSurvey2B1_all(ddata, input_shape=(16, 512, 256), transform_type='complex',
                                      transform_type_target="imag")
    a, b = DG.__getitem__(0)
    model_obj = XXNet()

    total_param = model_obj.parameters()

    optim_obj = torch.optim.SGD(params=total_param, lr=0.002)
    loss_obj = torch.nn.L1Loss()

    input_tensor = torch.as_tensor(a[np.newaxis])
    target_tensor = torch.as_tensor(b[np.newaxis])
    pred_model = model_obj(input_tensor)
    loss_tensor = loss_obj(pred_model, target_tensor)
    optim_obj.zero_grad()
    loss_tensor.backward()
    optim_obj.step()

    plt.close('all')
    # Visualization of the result
    hplotf.plot_3d_list(pred_model.detach().numpy(), title='result')
    hplotf.plot_3d_list(target_tensor.detach().numpy(), title='target')
    hplotf.plot_3d_list(input_tensor.detach().numpy(), title='input')

