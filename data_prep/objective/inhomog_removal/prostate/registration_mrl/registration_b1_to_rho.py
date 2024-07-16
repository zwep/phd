"""
Woo we can start the registration process.
"""

import h5py
print("Loaded h5")
import re
print("Loaded re")
import helper.misc as hmisc
print("Loaded misc")
import os
print("Loaded os")
import data_prep.registration.RegistrationProcess as RegistrationProcess
print("Loaded reg")
import sys

"""
Create paths
"""
remote = True

if remote:
    ddata_b1p = '/local_scratch/sharreve/flavio_data'
    # These contain a variable amount of 3D scans.....
    ddata_rho = '/local_scratch/sharreve/mri_data/prostate_h5'
    dmask_rho = '/local_scratch/sharreve/mri_data/mask_h5'
    dest_dir = '/local_scratch/sharreve/mri_data/registrated_h5'
else:
    ddata_b1p = '/home/bugger/Documents/data/test_clinic_registration/flavio_data'
    # These contain a variable amount of 3D scans.....
    ddata_rho = '/home/bugger/Documents/data/test_clinic_registration/mri_data/prostate_h5'
    dmask_rho = '/home/bugger/Documents/data/test_clinic_registration/mri_data/mask_h5'
    dest_dir = '/home/bugger/Documents/data/test_clinic_registration/registrated_h5'


hmisc.create_datagen_dir(dest_dir, type_list=('test', 'train', 'validation'),
                         data_list=('input', 'target', 'mask', 'target_clean'))


"""
Create all the file combinations, already do the train/test/validaiton split
"""

train_perc = 0.70
validation_perc = 0.10
test_perc = 0.20
filter_field_strength = 'MRL'
list_prostate_patient = []
# List of number of prostate_mri_mrl patients
for d, subdir, f in os.walk(ddata_rho):
    if filter_field_strength in subdir:
        patient_id = os.path.basename(d)
        list_prostate_patient.append(patient_id)

list_prostate_patient = sorted(list_prostate_patient)
list_b1_field = sorted([os.path.join(ddata_b1p, x) for x in os.listdir(ddata_b1p) if x.endswith('mat')])

num_prostate = len(list_prostate_patient)
num_b1 = len(list_b1_field)

n_train_prostate = int(num_prostate * train_perc)
n_validate_prostate = int(num_prostate * validation_perc)
n_test_prostate = int(num_prostate * test_perc)
n_train_b1 = int(num_b1 * train_perc)
n_validate_b1 = int(num_b1 * validation_perc)
n_test_b1 = int(num_b1 * test_perc)

list_prostate_train = list_prostate_patient[0:n_train_prostate]
list_prostate_validation = list_prostate_patient[n_train_prostate: (n_train_prostate + n_validate_prostate)]
list_prostate_test = list_prostate_patient[-n_test_prostate:]

list_b1_train = list_b1_field[0:n_train_b1]
list_b1_validation = list_b1_field[n_train_b1: (n_train_b1 + n_validate_b1)]
list_b1_test = list_b1_field[-n_test_b1:]

"""
Execute for... validation now..
"""

for data_type in ['train', 'test', 'validation']:
# # data_type = 'validation'
# data_type = 'train'
# # data_type = 'train'
    if data_type == 'train':
        selected_prostate_list = list_prostate_train
        selected_b1_list = list_b1_train
    elif data_type == 'test':
        selected_prostate_list = list_prostate_test
        selected_b1_list = list_b1_test
    elif data_type == 'validation':
        selected_prostate_list = list_prostate_validation
        selected_b1_list = list_b1_validation

    # Locally.. we only have one, so we continue with the test file set...
    # Now get the files from the chosen MRI patients...
    max_acq_per_patient = 1
    for i_patient in selected_prostate_list:
        print('Processing patient ', i_patient)
        patient_path = os.path.join(ddata_rho, i_patient, filter_field_strength)
        patient_file_list = [os.path.join(patient_path, x) for x in os.listdir(patient_path) if x.endswith('h5')]
        # Filter here on which files we would like to maintain.
        # Doing ALL the files of ALL the patients is just too much
        # So we limit the data to 'max acq per patient'
        # We want to select those shapes with the largest number of slices...
        # DO WE??
        import helper.plot_class as hplotc
        import helper.array_transf as harray
        import numpy as np
        patient_file_shape_list = []
        for i_file in patient_file_list:
            with h5py.File(i_file, 'r') as h5_obj:
                temp_shape = h5_obj['data'].shape
                n_slice = temp_shape[0]
                temp_array = np.array(h5_obj['data'])
            n_step_size = int(0.10 * n_slice)
            temp_mask = np.diff([harray.get_treshold_label_mask(x).sum() for x in temp_array[::n_step_size]])
            mean_difference = np.mean(np.abs(temp_mask))
            patient_file_shape_list.append((mean_difference, i_file))
        # Now we sort this array.. and reverse the ordering to decensing
        temp_sel = sorted(patient_file_shape_list, key=lambda x: x[0])[:max_acq_per_patient]
        sel_patient_file_list = [x[-1] for x in temp_sel]
        # Mask array list has the same names.. just a different sub-directory name.
        sel_patient_mask_file_list = [re.sub('prostate_h5', 'mask_h5', x) for x in sel_patient_file_list]
        print('Using the following files: ')
        print('patient files')
        for i in sel_patient_file_list:
            print(f'\t{i}')
        print('mask files')
        for i in sel_patient_mask_file_list:
            print(f'\t{i}')
        print('b1 list')
        for i in selected_b1_list:
            print(f'\t{i}')

        # sel_b1_file = selected_b1_list[0]
        for sel_b1_file in selected_b1_list:
            regproc_obj = RegistrationProcess.RegistrationProcess(patient_files=sel_patient_file_list,
                                                                  patient_mask_files=sel_patient_mask_file_list,
                                                                  b1_file=sel_b1_file,
                                                                  dest_path=dest_dir,
                                                                  data_type=data_type,
                                                                  display=False,
                                                                  registration_options='affine;rigid',
                                                                  n_cores=16)

            regproc_obj.run()
