"""
Ugh.. niftis..
Gimme some PNGs
"""

import os
import helper.misc as hmisc
import helper.plot_class as hplotc
import matplotlib.pyplot as plt
import re
import helper.array_transf as harray
import numpy as np
from objective_configuration.inhomog_removal import MODEL_DIR, get_path_dict
import argparse
from loguru import logger
import sys
import skimage


parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, help='Provide the name of the directory that we want to post process')
parser.add_argument('-dataset', type=str, default='all',
                    help='Provide the name of the dataset on which we want to evaluate: '
                         'synthetic, 3T, patient, volunteer')
parser.add_argument('-n', type=str, help='Determines the number of files we are converting', default=10)

p_args = parser.parse_args()
path = p_args.path
dataset = p_args.dataset
n_files = int(p_args.n)

dconfig = os.path.join(MODEL_DIR, path)
path_dict, dataset_list = get_path_dict(dconfig)

if dataset == 'all':
    sel_dataset_list = dataset_list
else:
    if dataset in dataset_list:
        sel_dataset_list = [dataset]
    else:
        logger.debug(f'Unknown dataset selected: {dataset}')
        sys.exit()

for i_dataset in sel_dataset_list:
    d_mask = os.path.join(path_dict[i_dataset]['dpred'], 'mask')
    for data_type in ['input', 'pred']:
        d = os.path.join(path_dict[i_dataset]['dpred'], data_type)
        listdir = os.listdir(d)
        file_list = [x for x in listdir if x.endswith('nii.gz')]
        dest_dir = d + '_PNG'
        dest_dir_hist = d + 'hist_PNG'
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        if not os.path.isdir(dest_dir_hist):
            os.makedirs(dest_dir_hist)
        logger.debug(f'Creating the following directories: {d}, {dest_dir}')
        # Just do max 10..
        for i_file in file_list[:n_files]:
            base_name = hmisc.get_base_name(i_file)
            file_path = os.path.join(d, i_file)
            mask_file_path = os.path.join(d_mask, i_file)
            logger.debug(f'{i_file}')
            dest_file_path = os.path.join(dest_dir, base_name + '.png')
            dest_file_path_hist = os.path.join(dest_dir_hist, base_name + '.png')
            loaded_array = hmisc.load_array(file_path)
            loaded_mask = hmisc.load_array(mask_file_path)
            # Change order because NIFITs
            loaded_array = loaded_array.T[:, ::-1, ::-1]
            loaded_mask = loaded_mask.T[:, ::-1, ::-1]
            n_slice = loaded_array.shape[0]
            sel_slice = n_slice // 2
            sel_img = loaded_array[sel_slice]
            sel_mask = loaded_mask[sel_slice]
            sel_mask = skimage.img_as_bool(sel_mask).astype(int)
            sel_mask = harray.shrink_image(sel_mask, 0)
            # Create image...
            fig_obj = hplotc.ListPlot(sel_img * sel_mask, ax_off=True)
            fig_obj.figure.savefig(dest_file_path, bbox_inches='tight', pad_inches=0.0)
            # Create histogram
            fig, ax = plt.subplots()
            _ = ax.hist((sel_img * sel_mask).ravel(), bins=256, range=(0, 255), density=True)
            fig.savefig(dest_file_path_hist)
            hplotc.close_all()
