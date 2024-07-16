"""
We got ourselves some new data.

Some preprocessing is alredy done by check_conversion_to_h5 file. Including:
    Sorting on:
        patient
        date
        acquisition

    Conversion to h5 format

Here we check the mask creation on first first, middle and last slice of each array
Create a jpeg image out of with the orignal to chekc the performance

"""

import h5py
import matplotlib
# We dont want to show anything

import helper.array_transf as harray
import helper.plot_class as hplotc
import os
import sys
import re
import numpy as np

ddir = '/local_scratch/sharreve/mri_data/validate_file_order'
validation_dir = '/local_scratch/sharreve/mri_data/validate_masking'

# Get all the directories with files....
file_dict = {}
for d, _, f in os.walk(ddir):
    MRI_MRL = os.path.basename(d)
    patient_id = os.path.basename(os.path.dirname(d))
    filter_list = [os.path.join(d, x) for x in f if x.endswith('.h5')]
    n_files = len(filter_list)
    if n_files == 0:
        print('For this one we have zero ', d)
    else:
        file_dict.setdefault(patient_id, {})
        file_dict[patient_id].setdefault(MRI_MRL, [])
        file_dict[patient_id][MRI_MRL] = filter_list


# Process each patient, per MRI/MRL, per file
for i_patient, temp_dict in file_dict.items():
    for i_mri_mrl, file_list in temp_dict.items():
        print('Processing ', i_patient, i_mri_mrl)
        target_path = os.path.join(validation_dir, i_patient, i_mri_mrl)
        if not os.path.isdir(target_path):
            os.makedirs(target_path)

        for i_file in file_list:

            file_name = os.path.basename(i_file)
            file_name_no_ext = os.path.splitext(file_name)[0]

            with h5py.File(i_file, 'r') as h5_obj:
                A = np.array(h5_obj['data'])

            n_slice, _, _ = A.shape
            first_array = A[0]
            middle_array = A[n_slice//2]
            last_array = A[-1]
            mask_overview = []
            for i_array in [first_array, middle_array, last_array]:
                mask_array = harray.get_treshold_label_mask(i_array, treshold_value=np.mean(i_array) * 0.5)
                mask_overview.append(i_array)
                mask_overview.append(mask_array)

            mask_overview = np.array(mask_overview)[None]
            fig_handle = hplotc.ListPlot(mask_overview, start_square_level=3)
            target_file_path = os.path.join(target_path, file_name_no_ext + '.jpeg')
            fig_handle.figure.savefig(target_file_path)
            hplotc.close_all()

