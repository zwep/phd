"""
Already a simple script to view eveything
"""

import scipy.io
import numpy as np
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import os

dir_data = '/media/bugger/MyBook/data/7T_data/cardiac/V9_13975'

for i_file in os.listdir(dir_data):
    temp_path = os.path.join(dir_data, i_file)
    A = scipy.io.loadmat(temp_path)['reconstructed_data']
    A = np.squeeze(np.array(A))
    print('Size of data', A.shape)
    hplotc.SlidingPlot(A.T)

import reconstruction.ReadCpx as read_cpx
