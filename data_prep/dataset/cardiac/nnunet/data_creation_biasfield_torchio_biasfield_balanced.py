"""
Since TorhcIO biasfield looks good.. we are going to create a mixture of the simulated biasfield and the torch biasf.
In a balanced way though

"""

import helper.misc as hmisc
import sys
import collections
import re
import shutil
import os
import argparse


def get_new_file_selection(ddata_img, n_reference):
    file_list = os.listdir(ddata_img)
    # Get all the patients
    #
    patient_list = [re.findall('(to_|^)([A-Z0-9a-z].*)_loc', x)[0][-1] for x in file_list]
    # Count how many occur
    counter_obj = collections.Counter(patient_list)
    len(counter_obj)
    new_file_list = []
    counter = 0
    while counter <= n_reference:
        # Add semi-random files per patient so that we have some variation on the b1-fields
        for patient_id, n_files in counter_obj.items():
            sel_index = counter % n_files
            sel_file = [x for x in file_list if patient_id in x][sel_index]
            new_file_list.append(sel_file)
            counter += 1
    return new_file_list[:n_reference]


def copy_new_files(new_files, ddata, ddest, label=False):
    for i_file in new_files:
        if label:
            i_file = re.sub('_0000', '', i_file)
        dsource = os.path.join(ddata, i_file)
        dtarget = os.path.join(ddest, i_file)
        shutil.copy(dsource, dtarget)

"""
We are going to copy specific files from the FULL dataset to a new task..
"""

parser = argparse.ArgumentParser()
parser.add_argument('-vendor', type=str)
p_args = parser.parse_args()
vendor = p_args.vendor


if vendor == 'A':
    ddata_1 = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task901_Biasfield_MM1_A_balanced'
    ddata_2 = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task905_Biasfield_MM1_A_balanced'
    ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task907_Biasfield_Biasfield_MM1_A_balanced'
    dreference_train = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task501_MM1_A/imagesTr'
    dreference_test = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task501_MM1_A/imagesTs'
elif vendor == 'B':
    ddata_1 = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task902_Biasfield_MM1_B_balanced'
    ddata_2 = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task906_Biasfield_MM1_B_balanced'
    ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task908_Biasfield_Biasfield_MM1_B_balanced'
    dreference_train = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task502_MM1_B/imagesTr'
    dreference_test = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task502_MM1_B/imagesTs'
else:
    print('Unknown vendor: ', vendor)
    sys.exit()

ddest_train_labels = os.path.join(ddest, 'labelsTr')
ddest_train_img = os.path.join(ddest, 'imagesTr')
ddest_test_labels = os.path.join(ddest, 'labelsTs')
ddest_test_img = os.path.join(ddest, 'imagesTs')

# For data 1 and 2 just copy 50% of it..
# Create destinations
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=())
for ddata in [ddata_1, ddata_2]:
    ddata_train_labels = os.path.join(ddata, 'labelsTr')
    ddata_train_img = os.path.join(ddata, 'imagesTr')
    ddata_test_labels = os.path.join(ddata, 'labelsTs')
    ddata_test_img = os.path.join(ddata, 'imagesTs')
    #
    n_train_reference = len(os.listdir(dreference_train))
    n_test_reference = len(os.listdir(dreference_test))

    new_train_img_files = get_new_file_selection(ddata_train_img, int(0.5*n_train_reference))
    new_test_img_files = get_new_file_selection(ddata_test_img, int(0.5*n_test_reference))

    copy_new_files(new_train_img_files, ddata_train_img, ddest_train_img)
    copy_new_files(new_test_img_files, ddata_test_img, ddest_test_img)
    copy_new_files(new_train_img_files, ddata_train_labels, ddest_train_labels, label=True)
    copy_new_files(new_test_img_files, ddata_test_labels, ddest_test_labels, label=True)
