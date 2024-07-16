# encoding: utf-8

import numpy as np
import os
import importlib

import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc

"""
Used to check prostateX data.
"""

import pydicom

file_dir_pat_0 = '/home/bugger/Documents/data/prostateX/PROSTATEx/ProstateX-0000/07-07-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-05711'
file_dir_list = sorted(os.listdir(file_dir_pat_0))[30:40]

len(f'Amount of dirs {len(file_dir_list)}')

for i_file_dir in file_dir_list:
    file_dir = os.path.join(file_dir_pat_0, i_file_dir)
    list_files = os.listdir(file_dir)
    len(f'Amount of files {len(list_files)} - {i_file_dir}')
    all_images = []

    for i_file in list_files:
        file_path = os.path.join(file_dir, i_file)
        temp_img = pydicom.read_file(file_path).pixel_array
        # print(f'Image shape {temp_img.shape}')
        all_images.append(temp_img)

    stacked_images = np.stack(all_images)[np.newaxis]
    hplotf.plot_3d_list(stacked_images, title=i_file_dir)