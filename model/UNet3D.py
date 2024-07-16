# encoding: utf-8

from torch import nn as nn
from model.Blocks import ConvBlock3D
from helper_torch.misc import activation_selector
import torch
import numpy as np
import helper_torch.misc as htmisc

"""

"""


class Unet3D(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, chans=32, num_pool_layers=4, padding=0, **kwargs):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.final_activation_name = kwargs.get('final_activation', 'identity')
        self.normalization_name = kwargs.get('normalization', 'identity')
        self.debug = kwargs.get('debug', False)
        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock3D(in_chans=in_chans, mid_chans=ch, out_chans=ch * 2,
                                                             padding=padding,
                                                             normalization=htmisc.module_selector(self.normalization_name))])
        ch *= 2
        self.pooling_layer = nn.MaxPool3d(kernel_size=2, stride=2)  # Has no learn able parameters

        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock3D(in_chans=ch, mid_chans=ch, out_chans=ch * 2, padding=padding,
                                                    normalization=htmisc.module_selector(self.normalization_name))]
            ch *= 2

        self.finest_conv = ConvBlock3D(in_chans=ch, mid_chans=ch, out_chans=ch * 2, padding=padding,
                                       normalization=htmisc.module_selector(self.normalization_name))
        ch *= 2

        self.up_sample_layers = nn.ModuleList()
        self.up_conv_layers = nn.ModuleList()
        for i in range(num_pool_layers):
            self.up_sample_layers += [nn.ConvTranspose3d(in_channels=ch, out_channels=ch, kernel_size=2,
                                                         padding=0, stride=2)]
            self.up_conv_layers += [ConvBlock3D(in_chans=ch + ch // 2, mid_chans=ch // 2, out_chans=ch // 2, padding=padding,
                                                normalization=htmisc.module_selector(self.normalization_name))]
            ch //= 2

        self.conv_final = nn.Conv3d(ch, out_chans, kernel_size=1, padding=0)
        self.final_activation = activation_selector(self.final_activation_name)

    def forward(self, input):
        stack = []
        output = input

        # Apply down-sampling layers
        counter = 0
        for layer in self.down_sample_layers:
            counter += 1
            output = layer(output)
            if self.debug:
                print('Downsampled layer ', output.shape)
            stack.append(output)
            output = self.pooling_layer(output)

        output = self.finest_conv(output)
        if self.debug:
            print('Finest conv ', output.shape)
        # Apply up-sampling layers
        counter = 0
        for up_layer, conv_layer in zip(self.up_sample_layers, self.up_conv_layers):
            downsample_layer = stack.pop()
            # Upscale
            output = up_layer(output)
            if self.debug:
                print('Upsample transpose ', output.shape)
            # Crop and copy
            down_shape = np.array(downsample_layer.shape[-3:])
            out_shape = np.array(output.shape[-3:])
            pos1 = np.array(down_shape) // 2 - np.array(out_shape) // 2
            pos2 = pos1 + np.array(out_shape)
            downsample_layer = downsample_layer[:, :, pos1[0]:pos2[0], pos1[1]:pos2[1], pos1[2]:pos2[2]]
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv_layer(output)
            if self.debug:
                print('Upsample conv ', output.shape)
            counter += 1

        output = self.conv_final(output)
        if self.debug:
            print('Final output ', output.shape)
        output_activation = self.final_activation(output)  # This is one of the changes I can make now..
        return output_activation


if __name__ == "__main__":

    # # # Testing 3D unet model
    import model.UNet as model_unet
    import torch
    import torch.nn as nn
    import importlib
    import helper.nvidia_parser as hnvidia
    import torchsummary
    index_gpu, p_gpu = hnvidia.get_free_gpu_id(claim_memory=0.8)
    device = torch.device("cuda:{}".format(str(index_gpu)) if torch.cuda.is_available() else "cpu")

    importlib.reload(model_unet)
    A = Unet3D(num_pool_layers=3, chans=8, in_chans=1, out_chans=1, debug=True, padding=1).float()
    B_tens = torch.as_tensor(np.random.rand(1, 1, 128, 128, 128)).float()

    with torch.no_grad():
        res = A.forward(B_tens)

    res.shape

    with torch.no_grad():
        torchsummary.summary(A, (1,  136, 136, 122))

    loss_obj = torch.nn.L1Loss()
    optim_obj = torch.optim.SGD(A.parameters(), lr=0.0001)
    optim_obj.zero_grad()

    a_tens = torch.as_tensor(np.random.rand(1, 3, 128, 128, 128)).float()
    torch_pred = A(a_tens)

    down_shape = a_tens.shape[-3:]
    out_shape = torch_pred.shape[-3:]
    pos1 = np.array(down_shape) // 2 - np.array(out_shape) // 2
    pos2 = pos1 + np.array(out_shape)
    downsample_layer = a_tens[:, :, pos1[0]:pos2[0], pos1[1]:pos2[1], pos1[2]:pos2[2]]
    loss = loss_obj(torch_pred, downsample_layer)

    optim_obj.zero_grad()
    loss.backward()
    optim_obj.step()


    # Check number of parameters
    ground_truth = 19069955
    s = 0
    t = 0
    for i in A.parameters():
        temp = np.prod(i.shape)
        s += temp
        print(s, i.shape)
        if len(i.shape) > 1:
            t += temp

    print(s, t, ground_truth, '\n', abs(s-ground_truth), '\t', abs(t-ground_truth))
