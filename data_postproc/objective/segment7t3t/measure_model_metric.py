import torch
from scipy.spatial.distance import directed_hausdorff
import sys
import time
import argparse
import json
import torch.nn as nn
import os
import numpy as np
import sklearn.metrics
import pathlib
import matplotlib.pyplot as plt
import nibabel
import helper.misc as hmisc
import pandas as pd
from objective_configuration.segment7T3T import CLASS_INTERPRETATION, COLOR_DICT, \
    COLOR_DICT_RGB, CMAP_ALL, MY_CMAP, get_path_dict

"""
This scripts calculated dice score over the model results stored in ./Results

Needs to use as little self-defined functions

It is assumed that we have some model results stored in 
`/data/cmr7t3t/cmr7t/Results`
or in 
'/data/cmr7t3t/acdc/acdc_processed/Results'

These are created with the nnunet_run.py command
"""

from helper.metric import dice_score


def check_update_store_scores(score_dict, path_score):
    if os.path.isfile(path_score):
        stored_scores = hmisc.load_json(path_score)
        stored_scores.update(score_dict)
    else:
        stored_scores = score_dict

    if new_dict:
        stored_scores = score_dict

    hmisc.write_json(stored_scores, path_score)


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str)
# Allows for a selection of models
parser.add_argument('-model', type=str, default=None)
# Allows for the label flip between RV and LV.
parser.add_argument('-flip_labels', type=str, default=False)
# Append results or create a new file..
parser.add_argument('-new', type=bool, default=False)
p_args = parser.parse_args()
dataset = p_args.dataset
model_selection = p_args.model
flip_labels = p_args.flip_labels
new_dict = p_args.new

path_dict = get_path_dict(dataset)
ddata_label = path_dict['dlabel']
ddata_model_results = path_dict['dresults']
#
path_hausdorf_scores = path_dict['dhausdorf']
path_dice_scores = path_dict['ddice']
path_jaccard_scores = path_dict['djaccard']
path_assd_scores = path_dict['dassd']


model_name_list = [x for x in os.listdir(ddata_model_results) if os.path.isdir(os.path.join(ddata_model_results, x))]
# Reverse sort, so that the last one that has been edited is equal to index 0
model_name_list = sorted(model_name_list, key=lambda x: os.path.getmtime(os.path.join(ddata_model_results, x)))[::-1]
model_name_list = np.array(model_name_list)
print("List of model names:")
for i, imodelname in enumerate(model_name_list):
    print(i, '\t', imodelname)

# Check which models we need to get
import objective_helper.segment7T3T as hsegm7t
if model_selection:
    sel_model_name_list = hsegm7t.model_selection_processor(model_selection, model_name_list)
else:
    print("Continue with all models")
    sel_model_name_list = model_name_list

# We use the label files.. since that is the limiting factor. (for the 7T data though)
label_file_list = os.listdir(ddata_label)
label_file_name_list = [hmisc.get_base_name(x) for x in os.listdir(ddata_label)]

dice_dict = {}
hausdorf_dict = {}
jaccard_dict = {}
assd_dict = {}
t0 = time.time()
n_models = len(sel_model_name_list)
for ii, model_sub_dir in enumerate(sel_model_name_list):
    delta_t = time.time() - t0
    print('Time elapsed ', delta_t)
    print('Expected remaining time ', delta_t * (n_models - ii))
    t0 = time.time()
    print('\n\nModel ', model_sub_dir)
    _ = dice_dict.setdefault(model_sub_dir, {})
    _ = hausdorf_dict.setdefault(model_sub_dir, {})
    _ = jaccard_dict.setdefault(model_sub_dir, {})
    _ = assd_dict.setdefault(model_sub_dir, {})

    model_result_dir = os.path.join(ddata_model_results, model_sub_dir)
    # Filtering on Nifti files..
    model_result_files = [x for x in os.listdir(model_result_dir) if x.endswith('nii.gz')]
    model_result_files_names = [hmisc.get_base_name(x) for x in model_result_files]

    # print('Model result names ', model_result_files_names)
    # print('Label names ', label_file_name_list)
    # Take the intersection of the names between the available labels and model results
    valid_names = list(set(model_result_files_names).intersection(set(label_file_name_list)))
    # print('Valid names ', valid_names)

    # Filter the model result files
    filtered_model_result_files = sorted([x for x in model_result_files if hmisc.get_base_name(x) in valid_names])
    filtered_label_files = sorted([x for x in label_file_list if hmisc.get_base_name(x) in valid_names])
    # print(model_result_files)
    # print(label_file_list)
    #  print(filtered_model_result_files)
    # print(filtered_label_files)
    # Loop over each file and calculate the dice score
    print('Number of labeled files found after filtering', len(filtered_label_files))

    for label_file, result_file in zip(filtered_label_files, filtered_model_result_files):
        # print('Label file ', label_file)
        # print('Result file ', result_file)
        _ = dice_dict[model_sub_dir].setdefault(label_file, {})
        _ = hausdorf_dict[model_sub_dir].setdefault(label_file, {})
        _ = jaccard_dict[model_sub_dir].setdefault(label_file, {})
        _ = assd_dict[model_sub_dir].setdefault(label_file, {})


        x_label_location = os.path.join(ddata_label, label_file)
        x_result_location = os.path.join(model_result_dir, result_file)

        # Since the initial data is of the shape (nx, ny, n_card) I assume these are as well
        label_array = hmisc.load_array(x_label_location)
        result_array = hmisc.load_array(x_result_location)

#        print('Label array ', label_array.shape)
 #       print('Result array ', result_array.shape)
        if label_array.shape != result_array.shape:
            print('Label array and result array not the same shape')
            print('Label shape', label_array.shape, 'Results shape ', result_array.shape)
            break

        n_card = label_array.shape[-1]
        n_classes = len(set(list(label_array.ravel())))
        for i_class in range(1, n_classes):
            _ = dice_dict[model_sub_dir][label_file].setdefault(i_class, {})
            _ = hausdorf_dict[model_sub_dir][label_file].setdefault(i_class, {})
            _ = jaccard_dict[model_sub_dir][label_file].setdefault(i_class, {})
            _ = assd_dict[model_sub_dir][label_file].setdefault(i_class, {})

        if flip_labels:
            # Switch the LV and RV classes...
            # class_interpretation = {'1': 'RV', '2': 'MYO', '3': 'LV'}
            RV_index = result_array == 3
            LV_index = result_array == 1
            result_array[RV_index] = 1
            result_array[LV_index] = 3

        # i_card/n_card can also refer to the location that we are dealing with..
        # Soo... we are going to take a difference range
        if n_card > 3:
            range_card = np.arange(n_card // 2 - 2, n_card // 2 + 2)
        else:
            range_card = range(n_card)

        for i_card in range_card:
            label_array_card = label_array[:, :, i_card]
            result_array_card = result_array[:, :, i_card]
            for i_class in range(1, n_classes):
                label_bin = label_array_card == i_class
                result_bin = result_array_card == i_class
                dice_score_card = dice_score(label_bin, result_bin)
                # We need to get the indices....
                # misc
                hausdorf_1 = directed_hausdorff(np.argwhere(label_bin), np.argwhere(result_bin))[0]
                hausdorf_2 = directed_hausdorff(np.argwhere(result_bin), np.argwhere(label_bin))[0]
                hausdorf_card = max(hausdorf_1, hausdorf_2)
                dice_score_card = np.round(dice_score_card, 3)
                hausdorf_card = np.round(hausdorf_card, 3)
                jaccard_card = jaccard_score = sklearn.metrics.jaccard_score(label_bin.ravel(), result_bin.ravel())
                # Not sure yet how to calculate this score.. There are some implementation difficulties
                assd_card = 0
                # Store them in a dictionary..
                dice_dict[model_sub_dir][label_file][i_class][f'phase_{str(i_card).zfill(2)}'] = dice_score_card
                hausdorf_dict[model_sub_dir][label_file][i_class][f'phase_{str(i_card).zfill(2)}'] = hausdorf_card
                jaccard_dict[model_sub_dir][label_file][i_class][f'phase_{str(i_card).zfill(2)}'] = jaccard_card
                assd_dict[model_sub_dir][label_file][i_class][f'phase_{str(i_card).zfill(2)}'] = assd_card


check_update_store_scores(dice_dict, path_dice_scores)
check_update_store_scores(hausdorf_dict, path_hausdorf_scores)
check_update_store_scores(jaccard_dict, path_jaccard_scores)
check_update_store_scores(assd_dict, path_assd_scores)