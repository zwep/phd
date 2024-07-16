import numpy as np
import h5py
import pandas as pd
import helper.misc as hmisc
import helper.plot_class as hplotc
import os
import objective_helper.reconstruction as hrecon
from reconstruction.ReadCpx import ReadCpx
from objective_helper.reconstruction import scan2direct_array, resize_array
from objective_configuration.reconstruction import DDATA

"""
The data for inference is not aligned properly.
Lets see if we can fix that

It seems that the data SHOULD be aligned. Somewhere the high time and fully sampled didnt go through the same pipeline..
--> This is because we didnt pick all the data from the scan. But from a combination between scan and data
--> This caused that the input (u.s.) and target (f.s.) to be processed by two different data pipelines sometimes
--> Which resulted in a different FOV, pixel spacing, etc.
 
 This file thus essentially preprocessed the inference data
 
"""


def cpx2direct(file_name):
    cpx_obj = ReadCpx(file_name)
    A = cpx_obj.get_cpx_img()
    # This makes the order (ny, nx, ncoil, nloc)
    A_swapped = np.moveaxis(np.squeeze(A), (0, 1), (-2, -1))
    A_direct = scan2direct_array(A_swapped)
    return A_direct

ddest = '/media/bugger/MyBook/data/7T_data/cardiac_radial_us_fs'
dcsv = os.path.join(ddest, 'us_fs_collection.csv')
dscan = '/media/bugger/MyBook/data/7T_scan/cardiac'

# These are the V-numbers that are available during testing...
# We need to limit to these three
v_number_test = ['V9_16936', 'V9_17067', 'V9_19531']

file_csv = pd.read_csv(dcsv)
vnumber_indicator = file_csv['vnumber'].apply(lambda x: x in v_number_test)
sel_file_csv = file_csv[vnumber_indicator]

for ii, irow in sel_file_csv.iterrows():
    # These are now .cpx files..
    base_name_under_sampled = hmisc.get_base_name(irow['filename_undersampled_files'])
    under_sampled = irow['filename_undersampled_files']
    # Fully sampled names
    base_name_fully_sampled = hmisc.get_base_name(irow['filename_fullysampled_files'])
    fully_sampled = irow['filename_fullysampled_files']
    # Define new file name
    new_file_name = f"{base_name_under_sampled[:19]}_{irow['vnumber']}_{irow['anatomy']}.h5"
    print(new_file_name)

    # Load..
    direct_target = cpx2direct(fully_sampled)
    n_card_full = direct_target.shape[0]
    direct_target = resize_array(direct_target)
    with h5py.File(os.path.join(ddest, 'target', new_file_name), 'w') as f:
        f.create_dataset('kspace', data=direct_target)

    del direct_target

    # Load..
    direct_input = cpx2direct(under_sampled)
    n_card_us = direct_input.shape[0]
    direct_input = direct_input[::n_card_us // n_card_full][: n_card_full]
    direct_input = resize_array(direct_input)
    with h5py.File(os.path.join(ddest, 'input', new_file_name), 'w') as f:
        f.create_dataset('kspace', data=direct_input)

    del direct_input



# # # Checking inference data...
import os
import helper.plot_class as hplotc
dd = '/media/bugger/MyBook/data/7T_data/cardiac_radial_us_fs'
dd_input = os.path.join(dd, 'input')
dd_sense = os.path.join(dd, 'sensitivity')
dd_targe = os.path.join(dd, 'target')

file_list = [x for x in os.listdir(dd_input) if x.endswith('h5')]


def file_to_dsp(img):
    sos_img = np.sqrt((np.abs(np.fft.ifftn(img[..., ::2] + 1j * img[..., 1::2], axes=(0, 1))) ** 2).sum(axis=-1))
    return sos_img


def kspace_to_dsp(img):
    # sos_img = np.sqrt((np.abs(img[..., 0] + 1j * img[..., 1])).sum(axis=0))
    sos_img = np.fft.ifftn(np.fft.ifftshift((img[..., 0::2] + 1j * img[..., 1::2]), axes=(0,1)), axes=(0,1))[..., 0]
    return sos_img


def sense_to_dsp(img):
    # sos_img = np.sqrt((np.abs(img[..., 0] + 1j * img[..., 1])).sum(axis=0))
    sos_img = (np.abs(img[..., 0] + 1j * img[..., 1]))[0]
    return sos_img


for i_file in file_list:
    input_path = os.path.join(dd_input, i_file)
    sense_path = os.path.join(dd_sense, i_file)
    tgt_path = os.path.join(dd_targe, i_file)

    input_sos = kspace_to_dsp(hmisc.load_array(input_path, data_key='kspace', sel_slice='mid'))
    sense_arr = sense_to_dsp(hmisc.load_array(sense_path, data_key='kspace', sel_slice='mid'))
    target_sos = kspace_to_dsp(hmisc.load_array(tgt_path, data_key='kspace', sel_slice='mid'))

    hplotc.ListPlot([input_sos, sense_arr, target_sos], ax_off=True, title=i_file)