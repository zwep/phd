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
parser.add_argument('-order', type=str, default='5')
parser.add_argument('-data', type=str)
p_args = parser.parse_args()
order = int(p_args.order)
data_type = p_args.data

"""

"""

if order == 5:
    ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task613_ACDC'
elif order == 2:
    ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task614_ACDC'
else:
    print('Unknown order')
    sys.exit()


ddata_acdc = f'/data/cmr7t3t/acdc/acdc_processed'
ddata_acdc_img_train = os.path.join(ddata_acdc, 'Image')
ddata_acdc_label_train = os.path.join(ddata_acdc, 'Label')
ddata_acdc_img_test = os.path.join(ddata_acdc, 'ImageTest')
ddata_acdc_label_test = os.path.join(ddata_acdc, 'LabelTest')

# Determine how to split everything into train/test/validation...
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=())

# # Now all the training and such has been written down

if data_type == 'train':
    file_list = sorted(os.listdir(ddata_acdc_img_train))
    ddata_acdc_img = ddata_acdc_img_train
    ddata_acdc_label = ddata_acdc_label_train
    ddest_img = os.path.join(ddest, 'imagesTr')
    ddest_label = os.path.join(ddest, 'labelsTr')
elif data_type == 'test':
    file_list = sorted(os.listdir(ddata_acdc_img_test))
    ddata_acdc_img = ddata_acdc_img_test
    ddata_acdc_label = ddata_acdc_label_test
    ddest_img = os.path.join(ddest, 'imagesTs')
    ddest_label = os.path.join(ddest, 'labelsTs')
else:
    print(f'Unknown data type {data_type}')
    sys.exit()


for i_file in file_list:
    base_name = hmisc.get_base_name(i_file)
    target_clean_file = os.path.join(ddata_acdc_img, i_file)
    target_segm_file = os.path.join(ddata_acdc_label, i_file)
    #
    affine_matrix = nibabel.load(target_clean_file).affine
    rho_array = hmisc.load_array(target_clean_file)
    rho_array = harray.scale_minmax(rho_array)
    segm_array = hmisc.load_array(target_segm_file)
    n_loc = rho_array.shape[-1]
    print('Number of locations ', n_loc)
    for i_loc in range(n_loc):
        sel_rho_array = rho_array[:, :, i_loc]
        sel_segm_array = segm_array[:, :, i_loc:i_loc+1]
        gen_biasf_obj = torchio.transforms.RandomBiasField(coefficients=0.8, order=order)
        gen_biasf = gen_biasf_obj(sel_rho_array[None, :, :, None])[0, :, :, 0]
        input_array = (sel_rho_array * gen_biasf)[:, :, None]
        input_array = harray.scale_minmax(input_array)
        input_nibabel_obj = nibabel.Nifti1Image(input_array, affine_matrix)
        target_nibabel_obj = nibabel.Nifti1Image(sel_segm_array[:, :, None], affine_matrix)
#
        nibabel.save(input_nibabel_obj, os.path.join(ddest_img, f'{base_name}_loc_{i_loc}_0000.nii.gz'))
        nibabel.save(target_nibabel_obj, os.path.join(ddest_label, f'{base_name}_loc_{i_loc}.nii.gz'))

