import os
import nibabel
import re
import numpy as np
import glob
import helper.misc as hmisc
import shutil
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-vendor', type=str)
p_args = parser.parse_args()
vendor = p_args.vendor

"""
Well we kinda already did this.. but lets re-create it...

"""

ddata_img = f'/data/cmr7t3t/mms1/all_phases_mid/Vendor_{vendor}/Image_single_slice'
ddata_label = f'/data/cmr7t3t/mms1/all_phases_mid/Vendor_{vendor}/Label_single_slice_swapped'
if vendor == 'A':
    ddest_train_labels = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task501_MM1_A/labelsTr'
    ddest_train_img = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task501_MM1_A/imagesTr'
    ddest_test_labels = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task501_MM1_A/labelsTs'
    ddest_test_img = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task501_MM1_A/imagesTs'
elif vendor == 'B':
    ddest_train_labels = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task502_MM1_B/labelsTr'
    ddest_train_img = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task502_MM1_B/imagesTr'
    ddest_test_labels = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task502_MM1_B/labelsTs'
    ddest_test_img = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task502_MM1_B/imagesTs'
else:
    print('Unknown vendor: ', vendor)
    sys.exit()

# Simply check one path.. if that one does not exist.. then create all the others..
# If it does exist, remove all the files. We want a clean start.
if os.path.isdir(ddest_train_labels):
    [os.remove(f) for f in glob.glob(ddest_train_img + '/*')]
    [os.remove(f) for f in glob.glob(ddest_train_labels + '/*')]
    [os.remove(f) for f in glob.glob(ddest_test_labels + '/*')]
    [os.remove(f) for f in glob.glob(ddest_test_img + '/*')]
else:
    os.makedirs(ddest_train_labels)
    os.makedirs(ddest_train_img)
    os.makedirs(ddest_test_labels)
    os.makedirs(ddest_test_img)

file_list = os.listdir(ddata_img)

patient_id = sorted(list(set([re.findall('([A-Z0-9a-z].*)_loc', x)[0] for x in file_list])))
n_patients = len(patient_id)
train_perc = 0.8
n_train = int(n_patients * train_perc)
train_patient_id = patient_id[:n_train]
train_file_list = []
test_file_list = []
print(f"Found {n_patients} number of patients")
print("Example patients ", patient_id[:10])
for i_patient in patient_id:
    i_patient_files = []
    for i, x in enumerate(file_list):
        if i_patient in x:
            i_patient_files = file_list.pop(i)
            if i_patient in train_patient_id:
                train_file_list.extend([i_patient_files])
            else:
                test_file_list.extend([i_patient_files])

print(f"Found {len(train_file_list)} number of train files")
print("Example train files ", train_file_list[:10])

print(f"Found {len(test_file_list)} number of test files")
print("Example test files ", test_file_list[:10])


for i_train in train_file_list:
    # Img
    base_name = hmisc.get_base_name(i_train)
    ext = hmisc.get_ext(i_train)
    dsource_file_img = os.path.join(ddata_img, i_train)
    ddest_file_img = os.path.join(ddest_train_img, base_name + "_0000" + ext)
    # Load array, check dimensions...
    loaded_array = hmisc.load_array(dsource_file_img)
    ndim = loaded_array.ndim
    if ndim == 2:
        # In this case, we add a third "z"-dimension
        loaded_array = loaded_array[:, :, None]
        nibabel_obj = nibabel.Nifti1Image(loaded_array, np.eye(4))
        nibabel.save(nibabel_obj, ddest_file_img)
    elif ndim == 3:
        # This should be fine, we can continue
        shutil.copy(dsource_file_img, ddest_file_img)
    else:
        print("Unknown number of dimensions ", ndim)
        continue
    # Label - similar process as the img...
    dsource_file_label = os.path.join(ddata_label, i_train)
    ddest_file_label = os.path.join(ddest_train_labels, i_train)
    # Load array, check dimensions...
    loaded_array = hmisc.load_array(dsource_file_label)
    ndim = loaded_array.ndim
    if ndim == 2:
        # In this case, we add a third "z"-dimension
        loaded_array = loaded_array[:, :, None]
        nibabel_obj = nibabel.Nifti1Image(loaded_array, np.eye(4))
        nibabel.save(nibabel_obj, ddest_file_label)
    elif ndim == 3:
        # This should be fine, we can continue
        shutil.copy(dsource_file_label, ddest_file_label)
    else:
        print("Unknown number of dimensions ", ndim)
        continue

# Same thing for test....
for i_test in test_file_list:
    # Img
    base_name = hmisc.get_base_name(i_test)
    ext = hmisc.get_ext(i_test)
    dsource_file_img = os.path.join(ddata_img, i_test)
    ddest_file_img = os.path.join(ddest_test_img, base_name + "_0000" + ext)
    # Load array, check dimensions...
    loaded_array = hmisc.load_array(dsource_file_img)
    ndim = loaded_array.ndim
    if ndim == 2:
        # In this case, we add a third "z"-dimension
        loaded_array = loaded_array[:, :, None]
        nibabel_obj = nibabel.Nifti1Image(loaded_array, np.eye(4))
        nibabel.save(nibabel_obj, ddest_file_img)
    elif ndim == 3:
        # This should be fine, we can continue
        shutil.copy(dsource_file_img, ddest_file_img)
    else:
        print("Unknown number of dimensions ", ndim)
        continue
    # Label - similar process as the img...
    dsource_file_label = os.path.join(ddata_label, i_test)
    ddest_file_label = os.path.join(ddest_test_labels, i_test)
    # Load array, check dimensions...
    loaded_array = hmisc.load_array(dsource_file_label)
    ndim = loaded_array.ndim
    if ndim == 2:
        # In this case, we add a third "z"-dimension
        loaded_array = loaded_array[:, :, None]
        # Is this eye(4) OK here...?
        nibabel_obj = nibabel.Nifti1Image(loaded_array, np.eye(4))
        nibabel.save(nibabel_obj, ddest_file_label)
    elif ndim == 3:
        # This should be fine, we can continue
        shutil.copy(dsource_file_label, ddest_file_label)
    else:
        print("Unknown number of dimensions ", ndim)
        continue