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
from model.XNet import XNetDown, XNetFeature, XNetUp

"""

"""


class XXXNet(torch.nn.Module):
    """
    Compsite model of the ones defined above...
    """
    def __init__(self, n_pool_layers=3, start_chan=4, n_hidden=512, **kwargs):
        super().__init__()
        output_activation = kwargs.get('output_activation', 'linear')
        convblock_activation = kwargs.get('convblock_activation', 'relu')
        feature_activation = kwargs.get('feature_activation', 'linear')

        self.n_concat = int((start_chan / 2) * 2 ** n_pool_layers * 8)
        print('Size of concatenation', self.n_concat)
        self.mod_down_real = XNetDown(chans=start_chan, num_pool_layers=n_pool_layers, convblock_activation=convblock_activation)
        self.mod_down_imag = XNetDown(chans=start_chan, num_pool_layers=n_pool_layers, convblock_activation=convblock_activation)
        self.mod_down_list = [self.mod_down_real, self.mod_down_imag]

        self.mod_mid_real = XNetFeature(self.n_concat, n_hidden, actv=True, feature_activation=feature_activation)
        self.mod_mid_imag = XNetFeature(self.n_concat, n_hidden, actv=True, feature_activation=feature_activation)
        self.mod_mid_list = [self.mod_mid_real, self.mod_mid_imag]

        # Used for ... mixing...
        # Ik kan deze dus ook ervoor doen..
        self.mod_mid_complex = XNetFeature(self.n_concat*2, n_hidden, actv=False, feature_activation=feature_activation)


        self.mod_up_real = XNetUp(in_chans=self.n_concat // 8 * 2, out_chans=1, num_pool_layers=n_pool_layers,
                                  output_activation=output_activation,
                                  convblock_activation=convblock_activation)
        self.mod_up_complex = XNetUp(in_chans=self.n_concat // 8 * 2, out_chans=1, num_pool_layers=n_pool_layers,
                                     output_activation=output_activation,
                                     convblock_activation=convblock_activation)

        self.mod_up_list = [self.mod_up_real, self.mod_up_complex]

    def forward(self, x):
        #print('INFO input shape', x.shape)
        input_tensor_real_imag = torch.split(x, 8, dim=1)
        # print('INFO split input len', len(input_tensor_real_imag))

        stack_compontent_list = []
        result_component_list = []
        final_result_component_list = []
        for i_compontent, x_compontent in enumerate(input_tensor_real_imag):
            # print('INFO component shape ', x_compontent.shape)
            input_tensor_list = torch.split(x_compontent, 1, dim=1)
            # print('INFO len component split ', len(input_tensor_list))
            output_stack_mod_down = [self.mod_down_list[i_compontent](x) for x in input_tensor_list]  # Model...
            result_down, stack_mod_down = zip(*output_stack_mod_down)
            cat_result_down = torch.cat(result_down, dim=1)
            # print('INFO concat down shape ', cat_result_down.shape)
            # Move for the dense layers that are coming up...
            cat_result_down_perm = cat_result_down.permute((0, 2, 3, 1))
            result_mid_perm = self.mod_mid_list[i_compontent](cat_result_down_perm)
            print('INFO ftr mid shape ', result_mid_perm.shape)

            stack_compontent_list.append(stack_mod_down)
            result_component_list.append(result_mid_perm)

        # En dan zou nu dan concat komen van beide delen...
        result_cat_complex = torch.cat(result_component_list, dim=-1)
        #print('INFO ftr mid shape ', result_cat_complex.shape)
        result_mid_complex = self.mod_mid_complex(result_cat_complex)
        mod_concat_derp = result_mid_complex.permute((0, 3, 1, 2))
        mod_concat_derp = torch.split(mod_concat_derp, self.n_concat, dim=1)
        #print('INFO mid_concat derp ', len(mod_concat_derp))
        #print('INFO mid_concat derp shape ', mod_concat_derp[0].shape)

        for i_compontent, x_compontent in enumerate(mod_concat_derp):
            result_mid_split = torch.split(x_compontent, self.n_concat // 8, dim=1)
          #  print('INFO result_mid_split ', len(result_mid_split))
         #   print('INFO result_mid_split shape ', result_mid_split[0].shape)
            stack_mod_down = stack_compontent_list[i_compontent]
            result_up = [self.mod_up_list[i_compontent](x, stack=y) for x, y in zip(result_mid_split, stack_mod_down)]  # Model...
            output = torch.cat(result_up, dim=1)
            final_result_component_list.append(output)

        output = torch.cat(final_result_component_list, dim=1)
        return output


if __name__ == "__main__":

    import data_generator.Rx2Tx as dg_rxtx
    from torch.utils.data import DataLoader
    import model.XXXNet
    import importlib
    importlib.reload(model.XNet)

    # # # Test XXX Net
    ddata = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels'
    DG = dg_rxtx.DataSetSurvey2B1_all(ddata, input_shape=(16, 512, 256), transform_type='complex')
    a, b = DG.__getitem__(0)
    model_obj = XXXNet()

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

