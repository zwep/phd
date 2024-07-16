import helper.plot_class as hplotc
import time
import pandas as pd
import matplotlib.pyplot as plt
import re
import helper.misc as hmisc
import numpy as np
from skimage.util import img_as_ubyte
import small_project.homogeneity_measure.metric_implementations as homog_measure
import helper.array_transf as harray
import os
from data_postproc.objective.segment7t3t.clustering.FileGatherer import FileGatherer
from objective_configuration.segment7T3T import DATASET_LIST, DATASET_SYNTH_LIST, CLASS_INTERPRETATION
from objective_configuration.segment7T3T import DICT_DATASET_DLABEL, DICT_DATASET_DIMG, DLOG
import argparse
import collections


class FeatureCalculator:
    def __init__(self, x_img, x_mask, patch_factor=4):
        self.x_img = x_img
        self.x_mask = x_mask
        self.patch_factor = patch_factor
#
    def calculate_intensity(self):
        # Based off the intensity, calculate some metrics
        temp_img = harray.scale_minmax(self.x_img)
        mean_intensity = np.mean(temp_img[self.x_mask])
        std_intensity = np.std(temp_img[self.x_mask])
        intensity_metric = {'mean': mean_intensity, 'std': std_intensity}
        return intensity_metric
#
    def calculate_hi(self):
        temp_img = harray.scale_minmax(self.x_img)
        # temp_mask = (self.x_mask > 0)
        hi_value = homog_measure.get_hi_value_integral(temp_img, self.x_mask)
        return {'hi': hi_value}
#
    def calculate_GLCM(self):
        temp_img = harray.get_crop(self.x_img, self.x_mask)[0]
        temp_img = harray.scale_minmax(temp_img)
        temp_img = img_as_ubyte(temp_img)
        patch_size = int(min(temp_img.shape) // self.patch_factor)
        glcm_patch_list = homog_measure.get_glcm_patch_object(temp_img, glcm_dist=[1, 2, 3, 5], patch_size=patch_size)
        glcm_features = homog_measure.get_glcm_features(glcm_patch_list)
        return glcm_features
#
    def calculate_fuzzy(self):
        temp_img = harray.get_crop(self.x_img, self.x_mask)[0]
        temp_img = harray.scale_minmax(temp_img)
        temp_img = img_as_ubyte(temp_img)
        patch_size = int(min(temp_img.shape) // self.patch_factor)
        fuzzy_features = homog_measure.get_fuzzy_features(temp_img, patch_size=patch_size)
        return fuzzy_features


dpng = '/home/bme001/20184098/visualization'


feature_dataset = {}
filegather_object_img = FileGatherer(debug=True, dict_dataset=DICT_DATASET_DIMG)
filegather_object_label = FileGatherer(debug=True, dict_dataset=DICT_DATASET_DLABEL)

# Copy the label files to the img object, since we have some mismatch between these two
# This is also the only dataset that suffers from it
base_name_label = [hmisc.get_base_name(x) for x in filegather_object_label.file_dict_list['7t']]
base_path_img = filegather_object_img.dataset_dir['7t']
temp_file_list = []
for ifile in filegather_object_label.file_dict_list['7t']:
    base_name = hmisc.get_base_name(ifile)
    base_ext = hmisc.get_ext(ifile)
    sel_file = os.path.join(base_path_img, base_name + "_0000" + base_ext)
    temp_file_list.append(sel_file)

filegather_object_img.file_dict_list['7t'] = temp_file_list

# This is to check whether all the files are the same...
for k, v in filegather_object_img.file_dict_list.items():
    print(f'---------- {k} ----------')
    label_file_names = [hmisc.get_base_name(x) for x in filegather_object_label.file_dict_list[k]]
    img_file_names = [hmisc.get_base_name(x) for x in v]
    for x, y in zip(label_file_names, img_file_names):
        print(x.strip() == y.replace('_0000', ''))


n_img = 100
for i_dataset in (DATASET_LIST + DATASET_SYNTH_LIST)[0:1]:
    t0 = time.time()
    print(i_dataset)
    dpng_dataset = os.path.join(dpng, i_dataset)
    if not os.path.isdir(dpng_dataset):
        os.makedirs(dpng_dataset)
    img_array = filegather_object_img.load_n_array(i_dataset, n_img)
    label_array = filegather_object_label.load_n_array(i_dataset, n_img)
    # Check if we have four classes in each array.
    label_ind = np.array([len(set(x.ravel().tolist())) == 4 for x in label_array])
    # Select only those where we have any valid classes..
    img_array = img_array[label_ind]
    label_array = label_array[label_ind].astype(int)
    # These are all the classes
    class_int_list = sorted(list(map(int, (set(label_array[0].ravel().tolist())))))
    # With this we can visualize everything..
    # plot_obj = hplotc.PlotCollage(content_list=img_array * label_array, ddest=dpng_dataset, plot_type='array', n_display=6)
    # plot_obj.plot_collage()
    # Count the amount of images we have left..
    sel_n_img = img_array.shape[0]
    file_feature_dict = {}
    for i_index in range(sel_n_img):
        file_path = filegather_object_label.file_dict_list[i_dataset][i_index]
        base_name = hmisc.get_base_name(file_path)
        _ = file_feature_dict.setdefault(base_name, {})
        sel_label = label_array[i_index]
        # collections.Counter(sel_label.ravel().tolist())
        sel_image = img_array[i_index]
        temp_class_dict = {}
        for i_class in [None] + class_int_list:
            if i_class is None:
                class_name = 'body'
            else:
                i_class_str = str(i_class // 85)
                class_name = CLASS_INTERPRETATION.get(i_class_str, 'All')
            _ = temp_class_dict.setdefault(class_name, {})
            # Now create the masked array
            if i_class is None:
                x_mask = np.ones(sel_label.shape, dtype=bool)
            else:
                if i_class == 0:
                    x_mask = (sel_label > 0).astype(bool)
                else:
                    x_mask = (sel_label == i_class)
            #
            x_img = sel_image * x_mask
            feature_calc_obj = FeatureCalculator(x_img=x_img, x_mask=x_mask)
            hi_dict = feature_calc_obj.calculate_hi()
            intensity_dict = feature_calc_obj.calculate_intensity()
            glcm_dict = feature_calc_obj.calculate_GLCM()
            fuzzy_dict = feature_calc_obj.calculate_fuzzy()
            temp_class_dict[class_name].update(hi_dict)
            temp_class_dict[class_name].update(intensity_dict)
            temp_class_dict[class_name].update(glcm_dict)
            temp_class_dict[class_name].update(fuzzy_dict)
            print(f'{i_index} / {n_img}', end='\r')
        file_feature_dict[base_name].update(temp_class_dict)
    # Now write the dataset away...
    print(f'Total time for dataset {i_dataset}', np.round(time.time() - t0, 2))
    dataset_features = harray.nested_dict_to_df(file_feature_dict)
    dataset_features = dataset_features.reset_index()
    dataset_features = dataset_features.rename({'0': 'values', 'level_0': 'filename',
                                                'level_1': 'class', 'level_2': 'feature'}, axis=1)
    dataset_features = dataset_features.pivot(columns='feature', values='values', index=['filename', 'class'])
    dataset_features.to_csv(os.path.join(dpng_dataset, 'image_feature.csv'))
