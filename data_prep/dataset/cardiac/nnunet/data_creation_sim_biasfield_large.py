import re
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

"""
With this script we can create a very big data set :)
"""


def store_n_items_from_data_generator(data_generator, n_items, file_suffix, flip_labels=False):
    for sel_item in range(n_items):
        cont = data_generator.__getitem__(sel_item % n_items)
        file_name = cont['file_name']  # This still has the extension .npy
        base_name = hmisc.get_base_name(file_name)
        #
        string_appendix = str(int(sel_item / len(data_generator)))
        #
        dest_img_file = os.path.join(ddest_img, f'{file_suffix}_{base_name}_{string_appendix}_0000.nii.gz')
        dest_label_file = os.path.join(ddest_label, f'{file_suffix}_{base_name}_{string_appendix}.nii.gz')
        #
        input = np.array(cont['input'])
        temp_target = np.array(cont['target'])
        # Convert segmentation from binary to integer
        x_padded = np.concatenate([np.zeros(temp_target.shape[-2:])[None], temp_target])
        x_rounded = np.isclose(x_padded, 1, atol=0.8).astype(int)
        target = np.argmax(x_rounded, axis=0)[np.newaxis]
        # Swap the labels. This is needed vor MM1
        if flip_labels:
            label_index_1 = target == 1
            label_index_3 = target == 3
            target[label_index_1] = 3
            target[label_index_3] = 1

        # Convert nibabel
        input_nibabel_obj = nibabel.Nifti1Image(input.T[::-1, ::-1], np.eye(4))
        target_nibabel_obj = nibabel.Nifti1Image(target.T[::-1, ::-1], np.eye(4))
        # Store the nibabel objects
        nibabel.save(input_nibabel_obj, dest_img_file)
        nibabel.save(target_nibabel_obj, dest_label_file)


ddata_mm1A = f'/data/cmr7t3t/biasfield_sa_mm1_A'
ddata_mm1B = f'/data/cmr7t3t/biasfield_sa_mm1_B'
ddata_acdc = '/data/cmr7t3t/biasfield_sa_acdc'

ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task631_Biasfield_MM1_A_MM1_B_ACDC'
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=[])

if datatype in ['test', 'train']:
    dataset_type = datatype  # Change this to train / test
else:
    print('Unknown data type. Exit program: ', datatype)
    sys.exit()

if dataset_type == 'train':
    n_creation = 10000
    ddest_img = os.path.join(ddest, 'imagesTr')
    ddest_label = os.path.join(ddest, 'labelsTr')
else:
    n_creation = 1000
    ddest_img = os.path.join(ddest, 'imagesTs')
    ddest_label = os.path.join(ddest, 'labelsTs')


dg_obj_mm1A = dg_segment_7t3t.DataGeneratorCardiacSegment(ddata=ddata_mm1A,
                                                          dataset_type=dataset_type,
                                                          target_type='segmentation',
                                                          transform_resize=True,
                                                          random_mask=True,
                                                          transform_type="abs")

dg_obj_mm1A.resize_list = [(256, 256)]
dg_obj_mm1A.resize_index = 0

dg_obj_mm1B = dg_segment_7t3t.DataGeneratorCardiacSegment(ddata=ddata_mm1B,
                                                          dataset_type=dataset_type,
                                                          target_type='segmentation',
                                                          transform_resize=True,
                                                          random_mask=True,
                                                          transform_type="abs")
dg_obj_mm1B.resize_list = [(256, 256)]
dg_obj_mm1B.resize_index = 0

dg_obj_acdc = dg_segment_7t3t.DataGeneratorCardiacSegment(ddata=ddata_acdc,
                                                          dataset_type=dataset_type,
                                                          target_type='segmentation',
                                                          transform_resize=True,
                                                          random_mask=True,
                                                          transform_type="abs")
dg_obj_acdc.resize_list = [(256, 256)]
dg_obj_acdc.resize_index = 0
# Filter only on one location such that we are aligned with the other locations of mm1
# file_list = dg_obj_acdc.container_file_info[0]['file_list']
# file_list = [x for x in file_list if 'loc_5' in x]
# dg_obj_acdc.container_file_info[0]['file_list'] = file_list


n_mm1a = len(dg_obj_mm1A)
n_mm1b = len(dg_obj_mm1B)
n_acdc = len(dg_obj_acdc)

print('Lengths of the datagenetors')
print(n_mm1a, n_mm1b, n_acdc)


# store_n_items_from_data_generator(dg_obj_mm1A, n_creation, file_suffix='mm1a')
store_n_items_from_data_generator(dg_obj_mm1B, n_creation, file_suffix='mm1b')
# store_n_items_from_data_generator(dg_obj_acdc, n_creation, file_suffix='acdc')
