"""
Now that we have manually selected ED and ES slices... we can continue
"""

import pandas as pd
import nibabel
import helper.plot_class as hplotc
import helper.misc as hmisc
import scipy.io
import helper.plot_class
import numpy as np
import os
import json

"""
Here we make a selection on which files we are going to use

Here we make use of the create JSON that filtered out the duplicated betweeen the old- and new-batch data
"""

ddata_nifti = '/data/cmr7t3t/cmr7t/RawData_newbatch/data_nifti'
ddata_ed_es = '/data/cmr7t3t/cmr7t/RawData_newbatch/data_nifti_ED_ES'
overview_dataframe = pd.read_csv('/data/cmr7t3t/cmr7t/overview_new_batch.csv')

# First delete the content, so that we dont have left-overs from previous changes
for f in os.listdir(ddata_ed_es):
    file_path = os.path.join(ddata_ed_es, f)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)


for ii, irow in overview_dataframe.iterrows():
    base_name = irow['mat files']
    subject_name = irow['subject name']
    iloc = int(irow['location'])
    visual_ind = irow['visual ok']
    previous_check = irow['not in previous']
    if previous_check:
        if visual_ind:
            nifti_file_name = subject_name + f'_{iloc}' + '.nii.gz'
            nifit_file = os.path.join(ddata_nifti, nifti_file_name)
            affine_struct = nibabel.load(nifit_file).affine
            data_obj = hmisc.load_array(nifit_file)
            nx, ny, ncard = data_obj.shape
            # Fix the ED slice
            ED_index = int(irow['ED slice'])
            ED_file_name = os.path.join(ddata_ed_es, "ED_" + subject_name + f'_{iloc}' + '.nii.gz')
            ED_slice_array = data_obj[:, :, ED_index:ED_index + 1]
            obj = nibabel.Nifti1Image(ED_slice_array, affine_struct)
            nibabel.save(obj, ED_file_name)
            # Fix the ES slice
            ES_index = int(irow['ES slice'])
            ES_file_name = os.path.join(ddata_ed_es, "ES_" + subject_name + f'_{iloc}' + '.nii.gz')
            ES_slice_array = data_obj[:, :, ES_index:ES_index + 1]
            obj = nibabel.Nifti1Image(ES_slice_array, affine_struct)
            nibabel.save(obj, ES_file_name)