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

import data_generator.UnetValidation as data_gen_unet

dir_data = '/home/bugger/Documents/data/grand_challenge/data'
res = data_gen_unet.UnetValidation(ddata=dir_data, input_shape=(1, 1, 1))
shape_list = []
for i in range(len(res.file_list)):
    res_A, res_B = res.__getitem__(i)
    shape_list.append((res_A.shape, res_B.shape))

input, target = zip(*shape_list)
# Input/Target stats
# min Z axis: id 20, 380x380x150
# min X/Y axis: id 36, 282x282x470
# max Z axis: id 59, 448 x 448 x 755
# max X/Y axis: id 67, 553 x 553 x 425