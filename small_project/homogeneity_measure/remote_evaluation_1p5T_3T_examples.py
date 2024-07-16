import json
import os
import h5py
import helper.array_transf as harray
import numpy as np
import small_project.homogeneity_measure.metric_implementations as homog_measure
"""
Make that comparisson with 3T and 1.5T data

"""

# Use the prostate_mri_mrl weighting data since they are mostly aligned
# Wait.. have I done the linking also based on slice number?...
# Lets check it

ddata = '/local_scratch/sharreve/mri_data/prostate_weighting_h5/train'
ddata_input = os.path.join(ddata, 'input')
ddata_input_cor = os.path.join(ddata, 'input_corrected')
ddata_target = os.path.join(ddata, 'target')
ddata_target_cor = os.path.join(ddata, 'target_corrected')

ddata_mask = os.path.join(ddata, 'mask')

ddata_list = [ddata_input, ddata_target, ddata_target_cor]
ddata_label_list = ['input', 'target', 'target_cor']
# ddata_label_list = ['target']
ddata_dict = dict(zip(ddata_label_list, ddata_list))

file_list = os.listdir(ddata_input)
feature_dict = {}
feature_dict.setdefault('hi', {})
for ddata_label in ddata_label_list:
    feature_dict['hi'].setdefault(ddata_label, [])

feature_dict.setdefault('fuzzy', {})
for ddata_label in ddata_label_list:
    feature_dict['fuzzy'].setdefault(ddata_label, [])

feature_dict.setdefault('glcm', {})
for ddata_label in ddata_label_list:
    feature_dict['glcm'].setdefault(ddata_label, [])

for sel_file in file_list:
    file_name, ext = os.path.splitext(sel_file)
    # Process selected file for all directories
    for ddata_label, ddata_dir in ddata_dict.items():
        print(ddata_label, sel_file)
        file_location = os.path.join(ddata_dir, sel_file)
        with h5py.File(file_location, 'r') as f:
            temp_img = np.array(f['data'])
        print('Shape', temp_img.shape)
        temp_img = harray.scale_minmax(temp_img)
        patch_size = int(0.1 * temp_img.shape[-1])
        stride = patch_size // 2
        if 'input' in ddata_label:
            mask_location = os.path.join(ddata_mask, file_name + "_input" + ext)
        else:
            mask_location = os.path.join(ddata_mask, file_name + "_target" + ext)
        with h5py.File(mask_location, 'r') as f:
            temp_mask = np.array(f['data'])
        if temp_mask.shape != temp_img.shape:
            print('Mask and img are not same shape. INFO: ', sel_file)
            print('\t\t File name ', sel_file)
            print('\t\t Data label ', ddata_label)
            print('\t\t Img shape ', temp_img.shape)
            print('\t\t Mask shape ', temp_mask.shape)
            # Will make a new mask... but this is strange... should not happen.
            temp_mask = np.array([harray.get_treshold_label_mask(x) for x in temp_img])
        hi_list = []
        luka_two_list = []
        contrast_list = []
        for x_mask, x in zip(temp_mask[::10], temp_img[::10]):
            hi_list.append(homog_measure.get_hi_value_integral(x, mask=x_mask))
            luka_two_list.append(homog_measure.get_fuzzy_luka_order(x, patch_size=patch_size, stride=stride, order=2))
            temp = homog_measure.get_glcm_patch_object(x, patch_size=patch_size, stride=stride)
            contrast_7T = homog_measure.get_glcm_features(temp, feature_keys=['contrast'])
            contrast_list.append(contrast_7T['contrast'])
        # Store the HI value...
        feature_dict['hi'][ddata_label].append(hi_list)
        # Store the fuzzy value...
        feature_dict['fuzzy'][ddata_label].append(luka_two_list)
        # Store the fuzzy value...
        feature_dict['glcm'][ddata_label].append(contrast_list)

ser_json_config = json.dumps(feature_dict)
temp_config_name = os.path.join('/local_scratch/sharreve/feature_dict_prostate_weighting.json')
with open(temp_config_name, 'w') as f:
    f.write(ser_json_config)
