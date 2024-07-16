"""

Slightly different from the validation thing.. because here we create 3D .h5 files from the masks...

Later these masks will be used for the registration process
"""

import h5py
import helper.array_transf as harray
import helper.plot_class as hplotc
import os
import numpy as np

ddir = '/local_scratch/sharreve/mri_data/prostate_h5'
dest_dir = '/local_scratch/sharreve/mri_data/mask_h5'

# Get all the directories with files....
file_dict = {}
for d, _, f in os.walk(ddir):
    MRI_MRL = os.path.basename(d)
    patient_id = os.path.basename(os.path.dirname(d))
    #if MRI_MRL == filter_on_field_strength:
    filter_list = [os.path.join(d, x) for x in f if x.endswith('.h5')]
    n_files = len(filter_list)
    if n_files != 0:
        file_dict.setdefault(patient_id, {})
        file_dict[patient_id].setdefault(MRI_MRL, [])
        file_dict[patient_id][MRI_MRL] = filter_list


# Process each patient, per MRI or MRL, and per file
for i_patient, temp_dict in file_dict.items():
    for i_mri_mrl, file_list in temp_dict.items():
        print('Processing ', i_patient, i_mri_mrl)
        target_path = os.path.join(dest_dir, i_patient, i_mri_mrl)
        if not os.path.isdir(target_path):
            os.makedirs(target_path)

        # We know that this only works accurately for transversal slices anyway...
        file_list_transversal = [x for x in file_list if 'transversal' in x]
        for i_file in file_list_transversal:
            file_name = os.path.basename(i_file)
            file_name_no_ext, _ = os.path.splitext(file_name)
            target_file_path = os.path.join(target_path, file_name)
            with h5py.File(i_file, 'r') as h5_obj:
                A = np.array(h5_obj['data'])

            mask_array = []
            for i_array in A:
                temp_mask = harray.get_treshold_label_mask(i_array, treshold_value=np.mean(i_array) * 0.5)
                mask_array.append(temp_mask)
            mask_array = np.array(mask_array).astype(bool)
            with h5py.File(target_file_path, 'w') as hf:
                hf.create_dataset('data', data=mask_array)


