import re
import torch
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
parser.add_argument('-data', type=str)
p_args = parser.parse_args()
datatype = p_args.data

ddata_mm1A = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task501_MM1_A'
ddata_mm1B = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task502_MM1_B'
ddata_acdc = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task511_ACDC'

ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task633_Biasfield_MM1_A_MM1_B_ACDC'
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=[])

if datatype in ['test', 'train']:
    dataset_type = datatype  # Change this to train / test
else:
    print('Unknown data type. Exit program: ', datatype)
    sys.exit()

if dataset_type == 'train':
    n_creation = 10000
    img_subdir = 'imagesTr'
    label_subdir = 'labelsTr'
else:
    n_creation = 1000
    img_subdir = 'imagesTs'
    label_subdir = 'labelsTs'


def store_n_items_with_random_biasfield(ddata, ddest, img_subdir, label_subdir, n_creation, biasf_order=5):
    if 'mm1_a' in ddata.lower():
        prefix = 'mm1a'
    elif 'mm1_b' in ddata.lower():
        prefix = 'mm1b'
    elif 'acdc' in ddata.lower():
        prefix = 'acdc'
    else:
        print('derp')
        sys.exit()
    counter = 0
    img_dir = os.path.join(ddata, img_subdir)
    label_dir = os.path.join(ddata, label_subdir)
    dest_img_dir = os.path.join(ddest, img_subdir)
    dest_label_dir = os.path.join(ddest, label_subdir)
    file_list = os.listdir(img_dir)

    n_files = len(file_list)
    while counter < n_creation:
        sel_index = counter % n_files
        counter_suffix = str(int(counter / n_files)).zfill(2)
        counter += 1
        sel_file_img = file_list[sel_index]
        sel_file_label = re.sub('_0000', '', sel_file_img)

        # Define input and output file names
        sel_file_img_path = os.path.join(img_dir, sel_file_img)
        sel_file_label_path = os.path.join(label_dir, sel_file_label)
        dest_sel_file_img_path = os.path.join(dest_img_dir, f"{prefix}_{counter_suffix}_" + sel_file_img)
        dest_sel_file_label_path = os.path.join(dest_label_dir, f"{prefix}_{counter_suffix}_" + sel_file_label)

        rho_array = hmisc.load_array(sel_file_img_path).T[:, ::-1, ::-1]
        label_array = hmisc.load_array(sel_file_label_path).T[:, ::-1, ::-1]
        n_loc = rho_array.shape[0]
        rho_array = rho_array[n_loc // 2][None]
        label_array = label_array[n_loc // 2][None]
        gen_biasf_obj = torchio.transforms.RandomBiasField(coefficients=0.8, order=biasf_order)
        gen_biasf = gen_biasf_obj(torch.from_numpy(rho_array[:, :, :, None].copy()))[0, :, :, 0].numpy()
        input_array = (rho_array * gen_biasf)[None]
        input_nibabel_obj = nibabel.Nifti1Image(input_array.T[::-1, ::-1], np.eye(4))
        target_nibabel_obj = nibabel.Nifti1Image(label_array.T[::-1, ::-1], np.eye(4))

        nibabel.save(input_nibabel_obj, dest_sel_file_img_path)
        nibabel.save(target_nibabel_obj, dest_sel_file_label_path)



store_n_items_with_random_biasfield(ddata_mm1A, ddest, img_subdir, label_subdir, n_creation)
store_n_items_with_random_biasfield(ddata_mm1B, ddest, img_subdir, label_subdir, n_creation)
store_n_items_with_random_biasfield(ddata_acdc, ddest, img_subdir, label_subdir, n_creation)

