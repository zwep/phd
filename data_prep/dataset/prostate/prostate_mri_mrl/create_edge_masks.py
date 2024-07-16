"""
Need these edge masks for the fat...
"""

import helper.array_transf as harray
import cv2
from skimage.util import img_as_ubyte, img_as_uint
import numpy as np
import cv2
import h5py
import helper.plot_class as hplotc
import os
import helper.misc as hmisc

ddata = '/local_scratch/sharreve/mri_data/registrated_h5'
for i_sub in ['test', 'train', 'validation']:
    ddata_sub = os.path.join(ddata, i_sub, 'mask')
    ddest = os.path.join(ddata, i_sub, 'mask_edge')
    if not os.path.isdir(ddest):
        os.makedirs(ddest)
    file_list = os.listdir(ddata_sub)
    for i_file in file_list:
        file_ext = hmisc.get_ext(i_file)
        file_name = hmisc.get_base_name(i_file)
        print("This file ", ddata_sub, i_file)
        file_path = os.path.join(ddata_sub, i_file)
        loaded_mask = hmisc.load_array(file_path)
        edge_mask = np.array([harray.get_edge_mask(x.astype(np.int16), outer_size=20, inner_size=40) for x in loaded_mask])
        with h5py.File(os.path.join(ddest, file_name + file_ext), 'w') as f:
            f.create_dataset('data', data=edge_mask)
