"""
We got some 3T data

Lets get a segmentation mask for this....

We know that the following slices are pretty well aligned

sel_slice_3T = 40
sel_slice_1p5T = 47

"""

import numpy as np
import helper.plot_class as hplotc
import h5py
import os

# I dont have anything for 3D mask stuff... Might be useful later in life

# Define all the image locations..
ddata_base = '/home/bugger/Documents/data/3T/prostate/prostate_weighting/test'
ddata_3T = os.path.join(ddata_base, 'target')
ddata_3T_segmentation = os.path.join(ddata_base, 'segmentation')

# Now create the segmentation
sel_file = os.listdir(ddata_3T)[0]

sel_prostate_file = os.path.join(ddata_3T, sel_file)
with h5py.File(sel_prostate_file, 'r') as f:
    prostate_array = np.array(f['data'])

sel_slice_3T = 40
mask_obj = hplotc.MaskCreator(prostate_array[sel_slice_3T])
# npy for now......
np.save(os.path.join(ddata_3T_segmentation, '7_MR.npy'), mask_obj.mask)