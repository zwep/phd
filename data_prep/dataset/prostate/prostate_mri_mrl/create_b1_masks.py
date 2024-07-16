""""
The masks that I used are of the body/prostate anatomy
I sometimes also need the B1 mask....

Lets get it
"""

import h5py
import numpy as np
import os
import helper.array_transf as harray
import helper.misc as hmisc

ddata = '/local_scratch/sharreve/mri_data/registrated_h5'
for i_sub in ['test', 'train', 'validation']:
    ddata_sub = os.path.join(ddata, i_sub, 'input')
    ddest = os.path.join(ddata, i_sub, 'mask')
    file_list = os.listdir(ddata_sub)
    for i_file in file_list:
        file_ext = hmisc.get_ext(i_file)
        file_name = hmisc.get_base_name(i_file)
        print("This file ", ddata_sub, i_file)
        file_path = os.path.join(ddata_sub, i_file)
        loaded_array = hmisc.load_array(file_path)
        print("Has this shape ", loaded_array.shape)
        sum_abs_array = np.abs(loaded_array[:, 0] + 1j * loaded_array[:, 1]).sum(axis=1)
        mask_array = np.array([harray.get_treshold_label_mask(x) for x in sum_abs_array])
        with h5py.File(os.path.join(ddest, file_name + "_b1" + file_ext), 'w') as f:
            f.create_dataset('data', data=mask_array)


# # # Copy the test masks to the test nifit folder
# # # Need to convert them from h5 to nii.gz
import nibabel
import re
ddata = '/local_scratch/sharreve/mri_data/registrated_h5/test/mask'
ddest = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/mask_b1'
mask_files_h5 = [x for x in os.listdir(ddata) if '_b1' in x]
for i_file in mask_files_h5:
    file_name = hmisc.get_base_name(i_file)
    file_name = re.sub('_b1', '', file_name)
    file_path = os.path.join(ddata, i_file)
    mask_array = hmisc.load_array(file_path)
    max_slice = mask_array.shape[0]
    # Store the full range of a single file so we can `pop` it later during evaluation
    slice_index = np.array(list(set([int(x) for x in np.linspace(0, max_slice - 1, 25)])))[::-1]
    mask_array = mask_array[slice_index].T[::-1, ::-1]
    nibabel_obj = nibabel.Nifti1Image(mask_array.astype(np.int8), np.eye(4))
    dest_file_path = os.path.join(ddest, file_name + ".nii.gz")
    print("From ", file_path)
    print("To ", dest_file_path, end='\n\n')
    nibabel.save(nibabel_obj, dest_file_path)