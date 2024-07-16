import re
import helper.array_transf as harray
import torchio.transforms
import sys
import numpy as np
import os
import nibabel
import data_generator.Segment7T3T as dg_segment_7t3t
import helper.plot_class as hplotc
import glob
import helper.misc as hmisc
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-vendor', type=str)
parser.add_argument('-data', type=str)
p_args = parser.parse_args()
vendor = p_args.vendor
data_type = p_args.data

"""

"""

if vendor == 'A':
    ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task605_Biasfield_MM1_A'
elif vendor == 'B':
    ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task607_Biasfield_MM1_B'
else:
    print('Unknown input. Exiting program')
    sys.exit()


ddata_mm1 = f'/data/cmr7t3t/mms1/all_phases_mid/Vendor_{vendor}'
ddata_mm1_img = os.path.join(ddata_mm1, 'Image_single_slice')
ddata_mm1_label = os.path.join(ddata_mm1, 'Label_single_slice')

mm1_file_names = sorted(os.listdir(ddata_mm1_img))
n_mm1 = len(mm1_file_names)

# Determine how to split everything into train/test/validation...
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=())

train_perc = 0.80
test_perc = 0.20

n_train_mm1 = int(n_mm1 * train_perc)
n_test_mm1 = int(n_mm1 * test_perc)

list_mm1_train = mm1_file_names[0:n_train_mm1]
list_mm1_test = mm1_file_names[-n_test_mm1:]

# # Now all the training and such has been written down

if data_type == 'train':
    file_list = list_mm1_train
    ddest_img = os.path.join(ddest, 'imagesTr')
    ddest_label = os.path.join(ddest, 'labelsTr')
elif data_type == 'test':
    file_list = list_mm1_test
    ddest_img = os.path.join(ddest, 'imagesTs')
    ddest_label = os.path.join(ddest, 'labelsTs')
else:
    print(f'Unknown data type {data_type}')
    sys.exit()

for i_file in file_list:
    base_name = hmisc.get_base_name(i_file)
    target_clean_file = os.path.join(ddata_mm1_img, i_file)
    target_segm_file = os.path.join(ddata_mm1_label, i_file)

    rho_array = hmisc.load_array(target_clean_file)
    segm_array = hmisc.load_array(target_segm_file)
    rho_array = harray.scale_minmax(rho_array)
    label_index_1 = segm_array == 1
    label_index_3 = segm_array == 3
    segm_array[label_index_1] = 3
    segm_array[label_index_3] = 1
    segm_array = segm_array[None]

    # Try to immitate the number of training items frmo biasfield data
    for ii in range(13):
        gen_biasf_obj = torchio.transforms.RandomBiasField(coefficients=0.8, order=5)
        gen_biasf = gen_biasf_obj(rho_array[None, :, :, None])[0, :, :, 0]
        input_array = (rho_array * gen_biasf)[None]
        input_array = harray.scale_minmax(input_array)
        input_nibabel_obj = nibabel.Nifti1Image(input_array.T[::-1, ::-1], np.eye(4))
        target_nibabel_obj = nibabel.Nifti1Image(segm_array.T[::-1, ::-1], np.eye(4))

        nibabel.save(input_nibabel_obj, os.path.join(ddest_img, f'{base_name}_{ii}_0000.nii.gz'))
        nibabel.save(target_nibabel_obj, os.path.join(ddest_label, f'{base_name}_{ii}.nii.gz'))
