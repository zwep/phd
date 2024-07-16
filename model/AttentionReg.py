# encoding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
import torch.nn as nn
import torch
# Denk dat ik het wel begrijp..


class dualAtt_24(nn.Module):

    # Source
    # https://github.com/DIAL-RPI/Attention-Reg
    def __init__(self):
        super().__init__()


        self.relu = nn.ReLU(inplace=True)
        self.conv3d_7 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.pathC_bn1 = nn.BatchNorm3d(64)


        self.conv3d_8 = nn.Conv3d(in_channels=16, out_channels=4, kernel_size=3, stride=(1, 1, 1), padding=1)

        self.conv3d_9 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.pathC_bn2 = nn.BatchNorm3d(1)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 6)

        """layers for path global"""
        self.path1_block1_conv = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path1_block1_bn = nn.BatchNorm3d(32)
        self.maxpool_downsample_pathGlobal11 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.path1_block2_conv = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path1_block2_bn = nn.BatchNorm3d(32)
        self.maxpool_downsample_pathGlobal12 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1,2,2), padding=1)
        self.path1_block3_NLCross = NLBlockND_cross(32)

        self.path2_block1_conv = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path2_block1_bn = nn.BatchNorm3d(32)
        self.maxpool_downsample_pathGlobal21 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.path2_block2_conv = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path2_block2_bn = nn.BatchNorm3d(32)
        self.maxpool_downsample_pathGlobal22 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1,2,2), padding=1)
        self.path2_block3_NLCross = NLBlockND_cross(32)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # total_start_time = time.time()
        x_path1 = torch.unsqueeze(x[:, 0, :, :, :], 1)
        x_path2 = torch.unsqueeze(x[:, 1, :, :, :], 1)

        """path global (attention)"""
        x_path1 = self.path1_block1_conv(x_path1)
        x_path1 = self.path1_block1_bn(x_path1)
        x_path1 = self.relu(x_path1)
        x_path1 = self.maxpool_downsample_pathGlobal11(x_path1)
        # print(x_path1.shape)

        x_path1 = self.path1_block2_conv(x_path1)
        x_path1 = self.path1_block2_bn(x_path1)
        x_path1 = self.relu(x_path1)
        x_path1_0 = self.maxpool_downsample_pathGlobal12(x_path1)
        # print(x_path1.shape)


        x_path2 = self.path2_block1_conv(x_path2)
        x_path2 = self.path2_block1_bn(x_path2)
        x_path2 = self.relu(x_path2)
        x_path2 = self.maxpool_downsample_pathGlobal21(x_path2)
        # print(x_path2.shape)

        x_path2 = self.path2_block2_conv(x_path2)
        x_path2 = self.path2_block2_bn(x_path2)
        x_path2 = self.relu(x_path2)
        x_path2_0 = self.maxpool_downsample_pathGlobal22(x_path2)
        # print(x_path2.shape)

        x_path1 = self.path1_block3_NLCross(x_path1_0, x_path2_0)
        x_path1 = self.relu(x_path1)

        x_path2 = self.path2_block3_NLCross(x_path2_0, x_path1_0)
        x_path2 = self.relu(x_path2)

        x_pathC = torch.cat((x_path1, x_path2), 1)

        """path combined"""
        x = x_pathC
        x = self.pathC_bn1(x)

        x = self.conv3d_7(x)
        x = self.relu(x)

        x = self.conv3d_8(x)
        x = self.relu(x)

        x = self.conv3d_9(x)
        x = self.pathC_bn2(x)

        x = x.view(x.size()[0], -1)
        x = self.relu(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        # time_cost = time.time() - total_start_time
        # print('1 whole cycle time cost {}s'.format(time_cost))
        # time.sleep(30)
        return x