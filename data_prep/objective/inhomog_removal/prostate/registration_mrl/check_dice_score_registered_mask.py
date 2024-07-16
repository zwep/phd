import os
import numpy as np
import h5py
import json
import re
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc

"""
I do not trust the registration process

One thing we can do as quality check is to validate how well the original body masks overlaps
with the newly registered mask

KEEP IN MIND that we cut off a part of the slices.. this needs to be done in this setting too
"""

# Check a single mask stuff...
data_type = 'test'
field_strength = 'MRL'
registrated_mask_dir = f'/local_scratch/sharreve/mri_data/registrated_h5/{data_type}/mask'
registrated_b1_dir = f'/local_scratch/sharreve/mri_data/registrated_h5/{data_type}/input'

mean_dice_result = {}
for i_file in os.listdir(registrated_mask_dir):
    print(f'Processing file {i_file}')
    reg_mask_file = os.path.join(registrated_mask_dir, i_file)
    with h5py.File(reg_mask_file, 'r') as f:
        A_reg = np.array(f['data'])
    # Now get the associated original mask...
    i_file_no_ext = os.path.splitext(i_file)[0]
    print('Regex on following file name ', i_file_no_ext)
    result_file = re.findall('to_([0-9]+)_MR_(.*)', i_file_no_ext)
    MR_id, MR_date = result_file[0]
    print('Processing MR ID ', MR_id, MR_date)
    rho_mask_file = f'/local_scratch/sharreve/mri_data/mask_h5/{MR_id}_MR/{field_strength}/{MR_date}.h5'
    if os.path.isfile(rho_mask_file):
        print('Is file: ', rho_mask_file)
        with h5py.File(rho_mask_file, 'r') as f:
            A_rho = np.array(f['data'])
        max_slice = A_rho.shape[0]
        n_min = int(max_slice * 0.3)
        n_max = max_slice - int(max_slice * 0.3)
        # Select only a couple of slices...
        A_rho_sel = A_rho[n_min:n_max]
        A_rho_transf = []
        for i_slice in A_rho_sel:
            temp_A, temp_A_mask = harray.get_center_transformation(i_slice, i_slice)
            A_rho_transf.append(temp_A_mask)
        A_rho_transf = np.array(A_rho_transf)
        A_reg = A_reg.astype(int)
        A_rho_transf = A_rho_transf.astype(int)
        res_dice = [hmisc.dice_metric(x, y) for x, y in zip(A_rho_transf, A_reg)]
        mean_dice = np.mean(res_dice)
        mean_dice_result.update({i_file: str(mean_dice)})

ser_json_config = json.dumps(mean_dice_result)
temp_config_name = os.path.join('/local_scratch/sharreve/mri_data', f'{data_type}_dice_dict.json')
with open(temp_config_name, 'w') as f:
    f.write(ser_json_config)
