import nibabel
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
import scipy.io
import helper.plot_class
import numpy as np
import os
import json
import skimage.transform as sktransform
import csv
import pandas as pd

"""
Now that we have manually selected the images and updated the overview table
we are going to create the niftis
"""

ddata_mat = '/data/cmr7t3t/cmr7t/RawData_newbatch/data_mat'
ddata_nifti = '/data/cmr7t3t/cmr7t/RawData_newbatch/data_nifti'
ddata_new_json = '/data/cmr7t3t/cmr7t/json_data_new_batch.json'
ddata_csv = '/data/cmr7t3t/cmr7t/overview_new_batch.csv'

csv_obj = pd.read_csv(ddata_csv)

# No idea why... but this is the thing we used...

for ii, irow in csv_obj.iterrows():
    base_name = irow['mat files']
    subject_name = irow['subject name']
    iloc = int(irow['location'])
    visual_ind = irow['visual ok']
    previous_check = irow['not in previous']
    if previous_check:
        if visual_ind:
            print(previous_check, visual_ind)
            i_mat_file = base_name + '.mat'
            mat_file = os.path.join(ddata_mat, i_mat_file)
            mat_obj = scipy.io.loadmat(mat_file)
            data_obj = np.squeeze(mat_obj['data'])
            nx, ny, ncard, nloc = data_obj.shape
            # print(f" Data shape {data_obj.shape}")
            i_nifti_file = subject_name + f'_{iloc}' + '.nii.gz'
            nifti_file_name = os.path.join(ddata_nifti, i_nifti_file)
            card_array = np.abs(data_obj[:, :, :, iloc])[::-1, ::-1, :]
            print(card_array.shape)
            # Acquisition resolution
            n_card = card_array.shape[-1]
            card_array = harray.scale_minmax(card_array)
            card_array = sktransform.resize(card_array, (256, 256, n_card), preserve_range=True, anti_aliasing=False)
            nifti_obj = nibabel.Nifti1Image(card_array, np.eye(4))
            nibabel.save(nifti_obj, nifti_file_name)
