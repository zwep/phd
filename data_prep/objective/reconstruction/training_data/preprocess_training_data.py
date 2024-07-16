import numpy as np
import pandas

import helper.misc as hmisc
import helper.plot_class as hplotc
import os
import h5py
from objective_helper.reconstruction import resize_array, scan2direct_array
from objective_configuration.reconstruction import DDATA
import pandas as pd


"""
We need to re-do the training data creation

First we try with one example

Final goal is to store it as an h5 file with dimensions (n_loc, ny, nx, 2 * ncoil)

Luckily we still have the npy's remotely. So we are going to process that there.
Immediatley create  a mixed dataset as well

"""

# We have selected that these V-numbers belong to the test set
# So that we can have other V-numbers for training
v_number_test = ['V9_16936', 'V9_17067', 'V9_19531']

# I think this csv is not needed yet
# dus_fs_csv = os.path.join(DDATA, 'us_fs_collection.csv')
dvnumber_csv = os.path.join(DDATA, 'vnumber_overview.csv')

# Create the target dir and its sub dirs
ddest = os.path.join(DDATA, 'mixed')
hmisc.create_datagen_dir(ddest, type_list=('test', 'validation_1', 'validation', 'train'), data_list=(['input']))

vnumber_csv = pd.read_csv(dvnumber_csv)
# First create the train/test/val-split
vnum_set = set(list(vnumber_csv['vnumber']))
n_data = len(vnum_set)
print(f"Number of V-numbers : {n_data}")  # Answer is 30
n_test = len(v_number_test)
n_val = int(n_data * 0.10)
n_train = int(n_data * 0.80)
# Actually get these things..
vnum_train_val_set = vnum_set.difference(set(v_number_test))
v_number_train = list(vnum_train_val_set)[:n_train]
v_number_val = list(vnum_train_val_set)[n_train:]

for i, irow in vnumber_csv.iterrows():
    vnumber = irow['vnumber']
    anatomy = irow['anatomy']
    dsearch = os.path.join(DDATA, f'radial_dataset_{anatomy}')
    if vnumber in v_number_train:
         substr = 'train'
    elif vnumber in v_number_val:
        substr = 'validation'
    elif vnumber in v_number_test:
        substr = 'test'
    else:
        substr = 'none'
        print(f"Uh oh {irow}")
    #
    ddest = os.path.join(DDATA, 'mixed', substr, 'input')
    found_files = hmisc.find_all_files_in_dir(irow['filename'], dir_name=dsearch, ext='npy')
    if len(found_files) > 0:
        for i_file in found_files:
            base_name = hmisc.get_base_name(i_file)
            dest_file_path = os.path.join(ddest, base_name + '.h5')
            #
            A = np.load(i_file)
            A = np.moveaxis(A, (0, 1), (-2, -1))
            A_direct = scan2direct_array(A)
            # Fft shift is required before resize array because of the k-space center...
            # Reverted the scan2direct to shifting the kspace to center
            # So no fftshift is required here
            A_direct_crop = resize_array(A_direct)
            # A_direct_crop = resize_array(np.fft.fftshift(A_direct, axes=(1, 2)))
            # A_direct_crop = np.fft.ifftshift(A_direct_crop, axes=(1, 2))
            # Create the data...
            with h5py.File(dest_file_path, 'w') as f:
                f.create_dataset('kspace', data=A_direct_crop.astype(np.float32))

    else:
        print('Noo')
        print(irow)
