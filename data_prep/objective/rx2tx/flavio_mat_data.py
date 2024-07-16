# encoding: utf-8

import numpy as np
import os
import importlib

import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc

from scipy.io import loadmat

orig_dir = '/home/bugger/Documents/data/simulation/flavio_mat'
dest_dir = '/home/bugger/Documents/data/simulation/flavio_npy'
hmisc.create_datagen_dir(dest_dir, data_list=('input', 'target', 'mask'))


file_list = sorted([x for x in os.listdir(orig_dir) if x.endswith('.mat')])
n_files = len(file_list)
train_ratio = 0.7
n_train = int(train_ratio * n_files)
train_files = file_list[:n_train]

for i_file in file_list:
    file_name, file_ext = os.path.splitext(i_file)
    print(i_file)
    if i_file in train_files:
        appendix_train = 'train'
    else:
        appendix_train = 'test'

    print('\t', appendix_train)
    orig_file = os.path.join(orig_dir, i_file)

    # Collect the data...
    temp_data = loadmat(orig_file)['Model']
    for i_name in temp_data.dtype.names:
        img_data = temp_data[i_name][0][0]
        if img_data.ndim > 2:
            img_data = np.moveaxis(img_data, -1, 0)

        if i_name == 'B1plus':
            appendix_source = 'target'
        elif i_name == 'B1minus':
            appendix_source = 'input'
        elif i_name == 'Mask':
            appendix_source = 'mask'
        else:
            continue

        print('\t', appendix_source)
        dest_file = os.path.join(dest_dir, appendix_train, appendix_source, file_name)
        print(f'\twriting {dest_file}')
        np.save(dest_file, img_data)

