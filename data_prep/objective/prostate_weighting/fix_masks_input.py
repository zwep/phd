import numpy as np
import h5py
import os
import helper.array_transf as harray

"""
I have messed up the mask creation... lets re-do this quickly
"""

data_type = ["test", "train", "validation"]
ddata = "/home/sharreve/local_scratch/mri_data/prostate_weighting_h5"

# For loop
for sel_data_type in data_type:
    input_path = os.path.join(ddata, sel_data_type, "input")
    mask_path = os.path.join(ddata, sel_data_type, "mask")
    input_file_list = os.listdir(input_path)

    # For loop
    for sel_input_file in input_file_list:
        sel_input_file_name, _ = os.path.splitext(sel_input_file)
        source_input_file = os.path.join(input_path, sel_input_file)
        dest_mask_file = os.path.join(mask_path, sel_input_file_name + "_input.h5")

        with h5py.File(source_input_file, 'r') as f:
            input_array = np.array(f['data'])

        mask_array = []
        for i_array in input_array:
            temp_mask = harray.get_treshold_label_mask(i_array, treshold_value=np.mean(i_array) * 0.5)
            mask_array.append(temp_mask)

        mask_array = np.array(mask_array).astype(bool)
        with h5py.File(dest_mask_file, 'w') as hf:
            hf.create_dataset('data', data=mask_array)
