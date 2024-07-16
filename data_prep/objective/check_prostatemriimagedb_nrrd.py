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
Used to check data from the prostate_mri_mrl mri image database
"""

import nrrd

data_dir = '/home/bugger/Documents/data/prostatemriimagedatabase'
file_dir_list = sorted([x for x in os.listdir(data_dir) if x.endswith('.nrrd')])

i_file = file_dir_list[0]
file_path = os.path.join(data_dir, i_file)

temp_data, temp_header = nrrd.read(file_path)
temp_data.shape
temp_data = np.moveaxis(temp_data, -1, 0)[np.newaxis]

hplotf.plot_3d_list(temp_data[:, 0:16])

np.save(os.path.join(data_dir, 'derp.npy'), temp_data[0, 0])
nrrd.writer.write(os.path.join(data_dir, 'derp.nrrd'), temp_data[0, 0])
import h5py
with h5py.File(os.path.join(data_dir, 'derp.h5py'), 'w') as f:
    f.create_dataset('test', data=temp_data[0, 0])
