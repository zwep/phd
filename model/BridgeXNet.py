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
from model.XNet import XNetFeature, XNetUp, XNetDown
import helper_torch.layers as hlayers


class XNetBridge(torch.nn.Module):
    """
    Compsite model of the ones defined above...
    """
    def __init__(self, n_pool_layers=3, start_chan=4, n_hidden=512, in_chans=8, **kwargs):
        super().__init__()
        output_activation = kwargs.get('output_activation', 'identity')
        convblock_activation = kwargs.get('convblock_activation', 'relu')
        feature_activation = kwargs.get('feature_activation', 'identity')

        device = kwargs.get('device', 'cpu')

        self.mod_mid_list = []
        n_concat = int((start_chan / 2) * 2 ** (n_pool_layers) * in_chans)

        if n_pool_layers > 2:
            for i in range(0, 3):
                n_concat_temp = int((start_chan / 2) * 2 ** (n_pool_layers - i) * in_chans)
                self.mod_mid_list.append(XNetFeature(n_concat_temp, n_concat_temp, feature_activation=feature_activation))
                for i_mod in self.mod_mid_list:
                    print('Putting model to device ', device, i_mod)
                    i_mod.to(device)
        else:
            self.mod_mid_list.append(XNetFeature(n_concat, n_hidden, feature_activation=feature_activation))

        self.mod_down = XNetDown(chans=start_chan, num_pool_layers=n_pool_layers, convblock_activation=convblock_activation)
        self.mod_up = XNetUp(in_chans=n_concat // in_chans * 2, out_chans=1, num_pool_layers=n_pool_layers,
                             output_activation=output_activation,
                             convblock_activation=convblock_activation)

        self.swap_to_last = hlayers.SwapAxes2D(to_channel_last=True)  # used before Dense layers
        self.swap_from_last = hlayers.SwapAxes2D(to_channel_last=False)  # used before Conv layers
        self.n_concat = n_concat
        self.n_pool_layers = n_pool_layers
        self.in_chans = in_chans

    def forward(self, x):
        #
        # layer 1 ... channels X                                    --> cat up3...
        #   layer 2 ... channels 2x         --> hidden          --> cat up2
        #       layer 3 ... channels 3x     --> hidden      --> cat up1
        #           layer 4 ... channels 4x --> hidden --> cat orig

        # Create X amount of inputs...
        input_tensor_list = torch.split(x, 1, dim=1)
        # Run model over X inputs
        output_stack_mod_down = [self.mod_down(x) for x in input_tensor_list]
        # Unpack results in final outcome and intermediate
        result_down, stack_mod_down = zip(*output_stack_mod_down)
        # print('before', len(result_down), len(stack_mod_down), len(stack_mod_down[0]))  #
        # Change order of intermediate output from [[1,2,3], [1,2,3],...] to [[1,1,1], [2,2,2],...]
        stack_mod_down = hmisc.change_list_order(stack_mod_down)
        # print('after', len(result_down), result_down[0].shape, len(stack_mod_down), len(stack_mod_down[0]))  #

        counter = 0
        new_stack_mod = []
        for i, i_stack in enumerate(stack_mod_down):  # Loop over all the model things..
            # print('mod_', counter)
            temp_stack = torch.cat(i_stack, dim=1)
            if i > (self.n_pool_layers-2):
                temp_stack = self.swap_to_last(temp_stack)
                temp_stack = self.mod_mid_list[counter](temp_stack)
                temp_stack = self.swap_from_last(temp_stack)
                counter += 1

            n_concat = temp_stack.shape[1]
            temp_stack = torch.split(temp_stack, n_concat//self.in_chans, dim=1)
            new_stack_mod.append(temp_stack)

        new_stack_mod = hmisc.change_list_order(new_stack_mod)

        # for x, y in zip(result_down, new_stack_mod):
        #     print(x.shape, len(y), y[0].shape)
        result_up = [self.mod_up(x, stack=y) for x, y in zip(result_down, new_stack_mod)]  # Model...
        output = torch.cat(result_up, dim=1)
        return output


if __name__ == "__main__":
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    model_obj = XNetBridge(start_chan=4, in_chans=8, convblock_activation='tanh')
    A = np.sin(np.random.normal(0, 4, size=(1, 8, 64, 64)))
    res_A = model_obj.forward(torch.as_tensor(A).float())

    fig, ax = plt.subplots(2)
    ax[0].imshow(A[0][0])
    ax[1].imshow(res_A[0][0].detach().numpy())
