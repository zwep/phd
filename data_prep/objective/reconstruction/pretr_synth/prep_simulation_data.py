import numpy as np
import os
import helper.misc as hmisc
import data_generator.Segment7T3T as dg_segment_7t3t
from objective_helper.reconstruction import resize_array, scan2direct_array
import h5py
import re


"""
Current folder is organized as...

input 
[2, 8, ny, nx]
contains B1- files

target
[2, 8, ny, nx]
contains B1+ files

target_clean
[ny, nx]
contains anatomy files


So simply... combine it. Turn to fft. Shift. And store as h5 kspace

Folder structure should become

./
    test
        input
    train
        input
    validation
        input
"""

ddata = '/home/sharreve/local_scratch/mri_data/cardiac_synth_7T/biasfield_sa_mm1_A'
ddest = '/home/sharreve/local_scratch/mri_data/cardiac_synth_7T/direct_synth/mixed'

ddest_train = '/home/sharreve/local_scratch/mri_data/cardiac_synth_7T/direct_synth/train/input'
ddest_test = '/home/sharreve/local_scratch/mri_data/cardiac_synth_7T/direct_synth/test/input'
ddest_validation = '/home/sharreve/local_scratch/mri_data/cardiac_synth_7T/direct_synth/validation/input'


# Create the destination (sub) folders)
hmisc.create_datagen_dir(ddest, data_list=['input'])

dataset_type = 'train'  # Change this to train / test
dg_obj = dg_segment_7t3t.DataGeneratorCardiacSegment(ddata=ddata,
                                                     dataset_type=dataset_type, target_type='segmentation',
                                                     transform_resize=True,
                                                     transform_type="complex",
                                                     presentation_mode=True)
dg_obj.resize_list = [(256, 256)]
dg_obj.resize_index = 0

# for sel_item in range(2):
for sel_item in range(len(dg_obj)):
    print(f'Creating {sel_item} / {len(dg_obj)}', end='\r')
    cont = dg_obj.__getitem__(sel_item)
    # Define the name
    file_name = dg_obj.container_file_info[0]['file_list'][sel_item]
    base_name = hmisc.get_base_name(file_name)
    dest_file = os.path.join(ddest_train, base_name + '.h5')
    # Define the data
    multi_coil_img = cont['target_clean'] * cont['b1p_shim'] * cont['b1m_array']
    A_swapped = np.moveaxis(multi_coil_img[None], (0, 1), (-1, -2))
    A_direct = scan2direct_array(A_swapped)
    # STore the data
    with h5py.File(dest_file, 'w') as f:
        f.create_dataset('kspace', data=A_direct.astype(np.float32))

print("Finished with training data creation")
"""
Now also do test
"""


for sel_dataset_type in ['test', 'validation']:
    print(f"Starting creation data : {sel_dataset_type}")
    dg_obj = dg_segment_7t3t.DataGeneratorCardiacSegment(ddata=ddata,
                                                         dataset_type='test', target_type='segmentation',
                                                         transform_resize=True,
                                                         transform_type="complex",
                                                         presentation_mode=True)
    file_list = dg_obj.container_file_info[0]['file_list']
    #
    # Split the test
    if sel_dataset_type == 'test':
        filtered_file_list = [x for x in file_list if x.startswith('V9')]
        sel_ddest = ddest_test
    else:
        filtered_file_list = [x for x in file_list if not x.startswith('V9')]
        sel_ddest = ddest_validation
    #
    dg_obj.container_file_info[0]['file_list'] = filtered_file_list
    #
    dg_obj.resize_list = [(256, 256)]
    dg_obj.resize_index = 0
    #
    # for sel_item in range(2):
    for sel_item in range(len(dg_obj)):
        print(f'Creating {sel_item} / {len(dg_obj)}', end='\r')
        cont = dg_obj.__getitem__(sel_item)
        # Define the name
        file_name = dg_obj.container_file_info[0]['file_list'][sel_item]
        base_name = hmisc.get_base_name(file_name)
        dest_file = os.path.join(sel_ddest, base_name + '.h5')
        # Define the data
        multi_coil_img = cont['target_clean'] * cont['b1p_shim'] * cont['b1m_array']
        A_swapped = np.moveaxis(multi_coil_img[None], (0, 1), (-1, -2))
        A_direct = scan2direct_array(A_swapped)
        # STore the data
        with h5py.File(dest_file, 'w') as f:
            f.create_dataset('kspace', data=A_direct.astype(np.float32))

