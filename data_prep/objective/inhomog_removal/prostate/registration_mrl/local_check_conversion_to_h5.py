"""
On a remote server we converted dicom data to .h5

This data has been transfered to my local PC. Here we check how it looks
"""

import helper.plot_class as hplotc
import h5py
import numpy as np
import os


ddata = '/home/bugger/Documents/data/1.5T/prostate_mri_mrl/4_MR/MRL'
list_files = os.listdir(ddata)


for sel_file in sorted(list_files):
    file_dir = os.path.join(ddata, sel_file)
    h5_obj = h5py.File(file_dir, 'r')
    A = np.array(h5_obj['data'])
    print('Shape of array ', A.shape, sel_file)

sel_index = list_files.index('20210104_0002.h5')

sel_file = list_files[2]
file_dir = os.path.join(ddata, sel_file)
h5_obj = h5py.File(file_dir, 'r')
A = np.array(h5_obj['data'])
n_slice = A.shape[0]
print('Shape of array ', A.shape, sel_file, int(n_slice * 0.3), n_slice - int(n_slice * 0.3))
hplotc.SlidingPlot(A)