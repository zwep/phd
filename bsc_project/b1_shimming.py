
"""
This script was used to show the first how-to on shimming
"""

import helper.plot_class as hplotc
import tooling.shimming.b1shimming_single as mb1
import helper.array_transf as harray

import os
import numpy as np

dir_data = '/media/bugger/MyBook/data/7T_data/cardiac/shimseries'
# Filter out some files.. they were accidently added to the .zip
list_files = [x for x in os.listdir(dir_data) if ('radial' not in x) and (x.endswith('npy'))]

# Get one file
sel_index = 0
sel_file = list_files[sel_index]
dir_file = os.path.join(dir_data, sel_file)

# Load it...
A = np.load(dir_file)
print('Shape of the file is... ', A.shape)
if A.ndim == 3:
    print('Wrong dimensions.. we should skip this file..')
    print('Dimensions should be.. n_c, n_c, n_x, n_y')

# How to do the shimming?

# First check what we are going to shim...
hplotc.ListPlot([A[0]], augm='np.abs')

# This is to make sure that the MaskCreator tool gets the right input
pre_mask = np.abs(A).sum(axis=0)
mask_handle = hplotc.MaskCreator(pre_mask)

# Initiate the shimming procedure
shim_proc = mb1.ShimmingProcedure(A[0], mask_handle.mask)
opt_shim, opt_value = shim_proc.find_optimum()
new_array = harray.apply_shim(A[0], cpx_shim=opt_shim)
hplotc.ListPlot([A[0].sum(axis=0), new_array], augm='np.abs')