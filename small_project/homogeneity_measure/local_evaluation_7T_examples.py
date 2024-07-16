import json
import pydicom
import re
import os
import h5py
import helper.array_transf as harray
import numpy as np
import small_project.homogeneity_measure.metric_implementations as homog_measure
import skimage.transform as sktransform
"""
Similar to 3T 1.5T script

Now for 7T
"""

# Using Daan Reesink zn data
# Easier to use this dir. Already has the masks in there
ddata = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Seb_pred'
ddata_mask = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter'
# Filter on those IDs that greater or equal to 10. Those files are single files. Easier to handle
patient_id_list = [x for x in os.listdir(ddata) if int(re.findall('7TMRI0([0-9]{2})', x)[0]) >= 10]

ddata_label_list = ['rho', 'bias', 'uncor']
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

for sel_patient_id in patient_id_list:
    print(sel_patient_id)
    ddata_patient = os.path.join(ddata, sel_patient_id)
    ddata_patient_mask = os.path.join(ddata_mask, sel_patient_id)
    # Vreemd. De naamgeving is soms echt apart in Daan_filter folder
    rho_file = [os.path.join(ddata_patient, x) for x in os.listdir(ddata_patient) if x.endswith('dcm')]
    rho_file = [x for x in rho_file if ('bias' in x) or ('uncorrected' in x) or ('rho') in x]
    mask_file = [os.path.join(ddata_patient_mask, x) for x in os.listdir(ddata_patient_mask) if x.endswith('npy')][0]
    print(len(rho_file), rho_file)
    print(mask_file)
    mask_array = np.load(mask_file)
    if mask_array.ndim == 3:
        n_slice = mask_array.shape[0]
        mask_array = sktransform.resize(mask_array,
                                        output_shape=(n_slice, 1024, 1024),
                                        anti_aliasing=False, preserve_range=True)
    else:
        mask_array = sktransform.resize(mask_array,
                                        output_shape=(1024, 1024),
                                        anti_aliasing=False, preserve_range=True)

    for sel_rho_file in rho_file:
        if ('bias' in sel_rho_file):
            dict_key = 'bias'
        elif ('uncorrected' in sel_rho_file):
            dict_key = 'uncor'
        elif ('rho') in sel_rho_file:
            dict_key = 'rho'

        rho_array = pydicom.read_file(sel_rho_file).pixel_array
        print('Shape rho array ', dict_key, rho_array.shape)
        print('Shape mask array ', dict_key, mask_array.shape)
        # THIS is really weird.
        starting_index = 0
        if (sel_patient_id == '7TMRI019') and (dict_key != 'rho'):
            starting_index = 1
        for ii, array_slice in enumerate(rho_array[starting_index:]):
            x = array_slice
            x = harray.scale_minmax(x)
            if mask_array.ndim == 3:
                x_mask = mask_array[ii]
            else:
                x_mask = mask_array
            patch_size = int(0.1 * x.shape[-1])
            stride = patch_size // 2
            temp_hi = homog_measure.get_hi_value_integral(x, mask=x_mask)
            temp_luka = homog_measure.get_fuzzy_luka_order(x, patch_size=patch_size, stride=stride, order=2)
            temp = homog_measure.get_glcm_patch_object(x, patch_size=patch_size, stride=stride)
            contrast_7T = homog_measure.get_glcm_features(temp, feature_keys=['contrast'])
            feature_dict['hi'][dict_key].append({sel_patient_id: temp_hi})
            feature_dict['fuzzy'][dict_key].append({sel_patient_id: temp_luka})
            feature_dict['glcm'][dict_key].append({sel_patient_id: contrast_7T['contrast']})

ser_json_config = json.dumps(feature_dict)
temp_config_name = os.path.join('/home/bugger/Documents/paper/homogeneity_index/metrics_on_7T.json')
with open(temp_config_name, 'w') as f:
    f.write(ser_json_config)

