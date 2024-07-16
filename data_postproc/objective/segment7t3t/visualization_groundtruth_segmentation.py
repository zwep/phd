import os
import argparse
import helper.misc as hmisc
import helper.plot_class as hplotc
from objective_configuration.segment7T3T import CLASS_INTERPRETATION, COLOR_DICT, \
    COLOR_DICT_RGB, CMAP_ALL, MY_CMAP, get_path_dict
import numpy as np

"""
Here I can the plot from the various models
"""

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str)
parser.add_argument('-flip_labels', type=str, default=False)
p_args = parser.parse_args()
dataset = p_args.dataset
flip_labels = p_args.flip_labels

path_dict = get_path_dict(dataset)
ddest_png = path_dict['dpng']
ddata_img = path_dict['dimg']
ddata_label = path_dict['dlabel']

dest_ground_truth = os.path.join(ddest_png, 'ground_truth')
if not os.path.isdir(dest_ground_truth):
    os.makedirs(dest_ground_truth)

file_list_segmentations = os.listdir(ddata_label)
file_list_segmentations = [x for x in file_list_segmentations if x.endswith('nii.gz')]
counter = -1
for sel_file in file_list_segmentations:
    counter += 1
    base_name = hmisc.get_base_name(sel_file)
    base_ext = hmisc.get_ext(sel_file)
    dest_path = os.path.join(dest_ground_truth, base_name + ".png")
    print("Processing ", sel_file)
    sel_img_path = os.path.join(ddata_img, base_name + "_0000" + base_ext)
    segm_file_path = os.path.join(ddata_label, sel_file)
    img_array = np.array(hmisc.load_array(sel_img_path)).T[:, ::-1, ::-1]
    segm_array = np.array(hmisc.load_array(segm_file_path)).T[:, ::-1, ::-1]
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