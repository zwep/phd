"""
We want to crop so that we see more of the heart

We are going to use the ED and ES slices to crop
"""

import pandas as pd
import scipy.io
import helper.plot_class
import json
import helper.plot_class as hplotc
import nibabel
import helper.misc as hmisc
import numpy as np
import os
import helper.array_transf as harray
import skimage.transform as sktransf

ddata_ed_es = '/data/cmr7t3t/cmr7t/RawData_newbatch/data_nifti_ED_ES'
ddata_crop = '/data/cmr7t3t/cmr7t/RawData_newbatch/data_nifti_ED_ES_crop'


# First delete the content, so that we dont have left-overs from previous changes
for f in os.listdir(ddata_crop):
    file_path = os.path.join(ddata_crop, f)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)


for i_file in os.listdir(ddata_ed_es):
    base_name = hmisc.get_base_name(i_file)
    base_ext = hmisc.get_ext(i_file)
    data_obj = hmisc.load_array(os.path.join(ddata_ed_es, i_file))
    rotated_array = data_obj.T[:, ::-1, ::-1]
    nx = rotated_array.shape[-1]
    crop_coords = harray.get_crop_coords_center(rotated_array.shape[-2:], width=int(0.80 * nx))
    cropped_array = np.array([harray.apply_crop(x, crop_coords) for x in rotated_array])
    cropped_array = cropped_array.T[::-1, ::-1]
    nchan = cropped_array.shape[-1]
    cropped_array = sktransf.resize(cropped_array, output_shape=(256, 256) + (nchan,), anti_aliasing=False)
    # Store this stuff
    dest_file_name = os.path.join(ddata_crop, i_file)
    obj = nibabel.Nifti1Image(cropped_array, np.eye(4))
    nibabel.save(obj, dest_file_name)
