# encoding: utf-8

import numpy as np
import os
import h5py
import nrrd
import re

"""
Going to concat a lot of data into h5 files..
"""



"""
Creation of h5 data from the prostatemriimagedatabase folder
"""

local_system = True

if local_system:
    dir_data = '/home/bugger/Documents/data/prostatemriimagedatabase'
    dest_data = '/home/bugger/Documents/data/prostatemriimagedatabase_h5'
else:
    dir_data = '/data/seb/prostatemriimagedatabase'
    dest_data = '/data/seb/prostatemriimagedatabase_h5'

file_list = [x for x in os.listdir(dir_data) if x.endswith('.nrrd')]

for i_file in file_list:
    file_dir = os.path.join(dir_data, i_file)

    i_file, _ = os.path.splitext(i_file)
    file_name = re.sub('\.', '_', i_file)
    dest_file = os.path.join(dest_data, file_name + '.h5')

    temp_data, temp_header = nrrd.read(file_dir)
    n_slice = temp_data.shape[-1]
    with h5py.File(dest_file, 'w') as f:
        temp_data = np.flipud(np.rot90(temp_data))
        temp_data = np.moveaxis(temp_data, -1, 0)
        f.create_dataset('data', data=temp_data)

"""
Creation of h5 data from the semireal input/target (left out validation.. since that was only one slice,,

"""

import re
import h5py
import numpy as np
import helper.misc as hmisc
import os


local_system = False

if local_system:
    dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation'
    dest_data = '/home/bugger/Documents/data/semireal/prostate_simulation_h5'
else:
    dir_data = '/data/seb/semireal/prostate_simulation'
    dest_data = '/data/seb/semireal/prostate_simulation_h5'


# This is a known parameter for myself.
# This is the amount of phase variations I created
n_phase_variations = 10
# This is also a known fixed parameter.. They are 256 x 256 the images
n_y = n_x = 256

hmisc.create_datagen_dir(dest_data, data_list=['input', 'target', 'mask'])

for dir_list, sub_dir_list, file_list in os.walk(dir_data):
    file_list = [x for x in file_list if x.endswith('npy')]

    if len(file_list):
        i_data = os.path.basename(dir_list)
        i_type = os.path.basename(os.path.dirname(dir_list))

        # We have missed the 'target 'thing// 
        if i_data == "target":
            target_dir = os.path.join(dest_data, i_type, i_data)
            file_list = os.listdir(dir_list)
            # First pre-select on all the transofmrations..
            # Get all the unique combinations
            res = sorted(set([re.sub('__.*', '', x) for x in file_list]))
            print('amount of groups', len(res))

            for i_group in res:
                print(f'\t Treating group {i_group}')
                sel_file_list = [x for x in file_list if i_group in x]

                n_files = len(sel_file_list)
                print('\t Amount of files ', n_files)

                # WHen we are dealing with masks... we only have slice data
                if i_data == 'mask':
                    n_slice = n_files
                    file_shape = (n_slice, n_y, n_x)
                    temp_data = np.empty(file_shape, dtype=np.complex)
                    # Create h5 file...
                    target_file_dir = os.path.join(target_dir, i_group) + '.h5'

                    # Create a list of all potentional slice numbres
                    # First.. create a list of all possible phases and slice nrs..
                    slice_list = []
                    for i_file in sel_file_list:
                        find_ind = re.findall('__([0-9]+)', i_file)
                        if find_ind:
                            slice_nr = int(find_ind[0])
                            slice_list.append(slice_nr)

                    slice_list = list(set(slice_list))

                    with h5py.File(target_file_dir, 'w') as f:
                        for i_file in sel_file_list:
                            # print(f'\t\t Treating file {i_file}')
                            file_dir = os.path.join(dir_list, i_file)
                            # Check if we have found what we want..
                            find_ind = re.findall('__([0-9]+)', i_file)

                            if find_ind:
                                slice_nr = int(find_ind[0])
                            else:
                                print('Found nothing, ', i_file, i_type)
                                break

                            A = np.load(file_dir)
                            index_slice_nr = slice_list.index(slice_nr)
                            temp_data[index_slice_nr] = A

                        f.create_dataset('data', data=temp_data)
                # Now we also have phase AND slice data
                else:
                    n_slice = n_files // n_phase_variations
                    file_shape = (n_phase_variations, n_slice, n_y, n_x)
                    temp_data = np.empty(file_shape, dtype=np.complex)
                    # Create h5 file...
                    target_file_dir = os.path.join(target_dir, i_group) + '.h5'

                    # First.. create a list of all possible phases and slice nrs..
                    slice_list = []
                    phase_list = []
                    for i_file in sel_file_list:
                        find_ind = re.findall('__([0-9]+)_phase_([0-9])', i_file)
                        if find_ind:
                            slice_nr, phase_nr = map(int, find_ind[0])
                            slice_list.append(slice_nr)
                            phase_list.append(phase_nr)

                    # THis creates an ordered list that now acts as a reference guide for the slices we find.
                    slice_list = list(set(slice_list))
                    phase_list = list(set(phase_list))

                    if (max(slice_list)+1) == n_slice:
                        print('\t\t\t Amount of slices == max iter number')

                    with h5py.File(target_file_dir, 'w') as f:
                        for i_file in sel_file_list:
                            # print(f'\t\t Treating file {i_file}')
                            file_dir = os.path.join(dir_list, i_file)
                            # Check if we have found what we want..
                            find_ind = re.findall('__([0-9]+)_phase_([0-9])', i_file)

                            if find_ind:
                                slice_nr, phase_nr = map(int, find_ind[0])
                            else:
                                print('Found nothing, ', i_file, i_type)
                                break

                            A = np.load(file_dir)

                            index_phase_nr = phase_list.index(phase_nr)
                            index_slice_nr = slice_list.index(slice_nr)
                            temp_data[index_phase_nr, index_slice_nr] = A

                        f.create_dataset('data', data=temp_data)
