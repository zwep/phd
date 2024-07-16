import json
import argparse
import helper.array_transf as harray
import helper.plot_class as hplotc
import nibabel
import os
import numpy as np
import torch
import helper.misc as hmisc
import sys
from PIL import ImageColor
from matplotlib.colors import ListedColormap
from objective_configuration.segment7T3T import CLASS_INTERPRETATION, COLOR_DICT, \
    COLOR_DICT_RGB, CMAP_ALL, MY_CMAP, get_path_dict

"""
Here I can the plot from the various models
"""

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, help='Provide any of the follow dataset names: 7T, ACDC, MM1A, MM1B, MM2, Kaggle')
parser.add_argument('-model', type=str, default=None, help='Provide a list of model indices as "0,4,2,6". Or task numbers by prepending with a t as "t501,502,610", or provide the model name itself by prepending with an r before the model name as "t510, rmode_name"')
parser.add_argument('-flip_labels', type=str, default=False, help='Flip the LV and RV labels in the visualization')
parser.add_argument('-h', '--help', action='help', type=str, default=False)
p_args = parser.parse_args()
dataset = p_args.dataset
model_selection = p_args.model
flip_labels = p_args.flip_labels

path_dict = get_path_dict(dataset)
ddata_model = path_dict['dresults']
ddest = path_dict['dpng']
ddata_img = path_dict['dimg']

model_name_list = [x for x in os.listdir(ddata_model) if os.path.isdir(os.path.join(ddata_model, x))]
model_name_list = sorted(model_name_list, key=lambda x: os.path.getmtime(os.path.join(ddata_model, x)))[::-1]
model_name_list = np.array(model_name_list)

print("List of model names:")
for i, imodelname in enumerate(model_name_list):
    print(i, '\t', imodelname)

import objective_helper.segment7T3T as hsegm7t
if model_selection:
    sel_model_name_list = hsegm7t.model_selection_processor(model_selection, model_name_list)
else:
    print("Please select a model first..")
    sys.exit()


for i_model_name_dir in sel_model_name_list:
    print("Segmentation directory ", i_model_name_dir)
    model_dir_path = os.path.join(ddata_model, i_model_name_dir)
    dest_model_dir_path = os.path.join(ddest, i_model_name_dir)
    if os.path.isdir(dest_model_dir_path) is False:
        os.makedirs(dest_model_dir_path)
    file_list_segmentations = os.listdir(model_dir_path)
    file_list_segmentations = [x for x in file_list_segmentations if x.endswith('nii.gz')]
    counter = -1
    for sel_file in file_list_segmentations:
        counter += 1
        base_name = hmisc.get_base_name(sel_file)
        base_ext = hmisc.get_ext(sel_file)
        dest_path = os.path.join(dest_model_dir_path, base_name + ".png")
        print("Processing ", sel_file)
        sel_img_path = os.path.join(ddata_img, base_name + "_0000" + base_ext)
        pred_segm_file_path = os.path.join(model_dir_path, sel_file)
        img_array = np.array(hmisc.load_array(sel_img_path)).T[:, ::-1, ::-1]
        segm_array = np.array(hmisc.load_array(pred_segm_file_path)).T[:, ::-1, ::-1]
        if flip_labels:
            # Switch the LV and RV classes...
            # class_interpretation = {'1': 'RV', '2': 'MYO', '3': 'LV'}
            RV_index = segm_array == 3
            LV_index = segm_array == 1
            segm_array[RV_index] = 1
            segm_array[LV_index] = 3

        print("Shape array ", img_array.shape)
        # print("Shape label array ", label_array.shape)
        print("Shape pred array ", segm_array.shape)
        print("Unique segm", list(set(segm_array.ravel())))
        n_slice = img_array.shape[0]
        img_array = img_array[n_slice // 2]
        segm_array = segm_array[n_slice // 2].astype(int)
        fig_obj = hplotc.ListPlot(img_array, ax_off=True)
        # Here we can plot segmentations the nice way
        plot_ax = fig_obj.ax_list[0]
        mask_values = segm_array == 0
        print("Cmap", CMAP_ALL)
        n_levels = list(set(segm_array.ravel()))
        segm_array = np.ma.masked_where(mask_values, segm_array)
        plot_ax.imshow(segm_array, cmap=MY_CMAP, alpha=0.4, vmin=(0, 4))
        hplotc.close_all()
        fig_obj.figure.savefig(dest_path, bbox_inches='tight', pad_inches=0.0)
