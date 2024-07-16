import nibabel.filebasedimages
import nibabel
import numpy as np
import helper.plot_class as hplotc
import shutil
import os
import helper.misc as hmisc

target_dir = '/media/bugger/MyBook/data/m&m/MnM_dataset'
hmisc.create_datagen_dir(target_dir, type_list=('test', 'validation', 'train'), data_list=('input', 'target'))

training_dir = '/media/bugger/MyBook/data/m&m/MnM-2/training/'
correct_files = []
for d, _, f in os.walk(training_dir):
    if len(f):
        filter_f = [x for x in f if x.endswith('nii.gz')]
        temp_list = [os.path.join(d, i_filter) for i_filter in filter_f]
        correct_files.append(temp_list)

n_patients = len(correct_files)
n_patients_training = int(0.80 * n_patients)
n_patients_validation = n_patients - n_patients_training

target_sub_dir = 'train'
for i_patient, file_list in enumerate(correct_files):
    print('PATIENT NUMBER ', i_patient, target_sub_dir)
    if i_patient > n_patients_training:
        target_sub_dir = 'validation'

    filter_file_list = [x for x in file_list if 'CINE' not in x and 'gt' not in x]
    for i_file in filter_file_list:
        file_name_nii, ext_gz = os.path.splitext(i_file)
        file_name, ext_nii = os.path.splitext(file_name_nii)
        base_name = os.path.basename(file_name)
        ground_truth_file = file_name + '_gt' + ext_nii + ext_gz

        dest_input_file = os.path.join(target_dir, target_sub_dir, 'input', base_name + ext_nii + ext_gz)
        dest_target_file = os.path.join(target_dir, target_sub_dir, 'target', base_name + "_gt" + ext_nii + ext_gz)

        print(i_patient, i_file, dest_input_file, dest_target_file)

        # Input
        shutil.copy(i_file, dest_input_file)
        shutil.copy(ground_truth_file, dest_target_file)


"""
Now fix the test / evaluation dataset
"""
target_dir = '/media/bugger/MyBook/data/m&m/MnM_dataset'
hmisc.create_datagen_dir(target_dir, type_list=('test', 'validation', 'train'), data_list=('input', 'target'))

testing_dir = '/media/bugger/MyBook/data/m&m/MnM-2/validation/'

correct_files = []
for d, _, f in os.walk(testing_dir):
    if len(f):
        filter_f = [x for x in f if x.endswith('nii.gz')]
        temp_list = [os.path.join(d, i_filter) for i_filter in filter_f]
        correct_files.append(temp_list)

target_sub_dir = 'test'
for i_patient, file_list in enumerate(correct_files):
    print('PATIENT NUMBER ', i_patient, target_sub_dir)

    filter_file_list = [x for x in file_list if 'CINE' not in x and 'gt' not in x]
    for i_file in filter_file_list:
        file_name_nii, ext_gz = os.path.splitext(i_file)
        file_name, ext_nii = os.path.splitext(file_name_nii)
        base_name = os.path.basename(file_name)
        dest_input_file = os.path.join(target_dir, target_sub_dir, 'input', base_name + ext_nii + ext_gz)

        # Input
        shutil.copy(i_file, dest_input_file)


