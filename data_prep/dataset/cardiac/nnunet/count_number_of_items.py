import os
import re
from nnunet.paths import nnUNet_raw_data

"""
Count the number of items in each Task directory

ACDC patient010_frame01_loc_3_0000.nii.gz
MM1A A0S9V9_loc_00_0000
MM1B A1D0Q7_loc_00_0000.nii.gz
Kaggle 13-frame000-slice003_0000.nii.gz
MM2 001_SA_ED_0000.nii.gz

"""

def print_dir_len(x, name):
    file_list = os.listdir(x)
    print(name, ' ' * (50 - len(name)), len(file_list))

task_list = os.listdir(nnUNet_raw_data)
for task_name in sorted(task_list):
    task_dir = os.path.join(nnUNet_raw_data, task_name)
    train_dir = os.path.join(task_dir, 'imagesTr')
    test_dir = os.path.join(task_dir, 'imagesTs')
    if os.path.isdir(train_dir):
        print_dir_len(train_dir, task_name + ' train')
    if os.path.isdir(test_dir):
        print_dir_len(test_dir, task_name + ' test')



#
# ddata_mm1_a = '/data/cmr7t3t/mms1/all_phases_mid/Vendor_A/Image_single_slice'
# file_list = os.listdir(ddata_mm1_a)
# patient_id = list(set([re.findall('([0-9A-Z]{6})_loc', x)[0] for x in file_list]))
# print(sorted(patient_id))
# n_patients = len(patient_id)
#
# print(f"Number of patients MM1-A {n_patients} / {len(file_list)}")  # -- 75
#
#
# ddata_acdc = '/data/cmr7t3t/acdc/acdc_processed/Image'
# file_list = os.listdir(ddata_acdc)
# patient_id = list(set([re.findall('(patient[0-9]{3})', x)[0] for x in file_list]))
# n_patients = len(patient_id)
# # patient001_frame12.nii.gz
# print(f"Number of patients ACDC {n_patients} / {len(file_list)}")  # -- 69
#
# # Dubbel check op biasfield data
#
# ddata_biasf_acdc = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task567_Biasfield_ACDC/imagesTr'
#
# file_list = os.listdir(ddata_biasf_acdc)
# patient_id = list(set([re.findall('(patient[0-9]{3})', x)[0] for x in file_list]))
# n_patients = len(patient_id)
# # patient001_frame12.nii.gz
# print(f"Number of patients Biasf+ACDC {n_patients} / {len(file_list)}")  # -- 63
#
#
# # Dubbel check op GAN data
#
# ddata_gan_acdc = '/data/cmr7t3t/results/ACDC_220121/seven2three_acdc_cut_NCE4_GAN2_np128_fe_211208/test_100/niftis/cmr3t2cmr7t'
#
# file_list = os.listdir(ddata_gan_acdc)
# patient_id = list(set([re.findall('(patient[0-9]{3})', x)[0] for x in file_list]))
# n_patients = len(patient_id)
# # patient001_frame12.nii.gz
# print(f"Number of patients GAN+ACDC {n_patients} / {len(file_list)}")  # -- 70