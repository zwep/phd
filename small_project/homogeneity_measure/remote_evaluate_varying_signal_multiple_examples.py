from scipy.stats import pearsonr
import json
import time
import torch
import helper.plot_class as hplotc
import helper.plot_fun as hplotf
import os
import objective.inhomog_removal.recall_inhomog_removal as recall_inhom
import biasfield_algorithms.N4ITK as get_n4itk
import matplotlib.pyplot as plt
import tooling.shimming.b1shimming_single as mb1_single
import helper.array_transf as harray
import h5py
import helper.misc as hmisc
import numpy as np
import scipy.io
import skimage.data
import skimage.transform
import skimage.metrics
import scipy.integrate

import small_project.homogeneity_measure.create_dummy_data as data_generator
import small_project.homogeneity_measure.metric_implementations as homog_metric
import json
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

"""
The varying signal examples are nice

But currently only created for one anatomy + B1 combination
This can be done better...

This needs to be done remote. Since there we have all the data..
"""


"""
Load Rho data (not registered to a specific b1+ map....)
"""

ddata = '/local_scratch/sharreve/mri_data/registrated_h5/train'
ddata_rho = os.path.join(ddata, 'target_clean')
ddata_b1m = os.path.join(ddata, 'input')
ddata_mask = os.path.join(ddata, 'mask')
ddata_b1p = os.path.join(ddata, 'target')
# Doesnt really matter which dir... same names everhwere
file_list = os.listdir(ddata_rho)


def run_stuff(ii):
    sel_file = file_list[ii]
    # Define the locations of hte files...
    rho_location = os.path.join(ddata_rho, sel_file)
    b1m_location = os.path.join(ddata_b1m, sel_file)
    b1p_location = os.path.join(ddata_b1p, sel_file)
    mask_location = os.path.join(ddata_mask, sel_file)
    with h5py.File(rho_location, 'r') as f:
        temp = f['data']
        n_slice = temp.shape[0]
        sel_slice = n_slice // 2
        A_rho = np.array(f['data'][sel_slice])
    with h5py.File(b1m_location, 'r') as f:
        A_b1m = np.array(f['data'][sel_slice])
    with h5py.File(b1p_location, 'r') as f:
        A_b1p = np.array(f['data'][sel_slice])
    with h5py.File(mask_location, 'r') as f:
        A_mask = np.array(f['data'][sel_slice])
    A_rho = harray.scale_minmax(A_rho)
    A_b1m = harray.scale_minmax(A_b1m, is_complex=True)
    A_b1p = harray.scale_minmax(A_b1p, is_complex=True)
    data_obj = data_generator.DummyVaryingSignalData(rho=A_rho, b1p=A_b1p, b1m=A_b1m, mask=A_mask)
    varying_signal_image = data_obj.create_varying_signal_maps()
    selected_mask = A_mask
    patch_size = int(0.1 * A_rho.shape[-1])
    stride = patch_size // 2
    metric_values = []
    for ii in range(len(varying_signal_image)):
        input_image = varying_signal_image[ii] * selected_mask
        ssim_target_input = skimage.metrics.structural_similarity(input_image, A_rho)
        hi_input = homog_metric.get_hi_value_integral(input_image, selected_mask)
        luka_two = homog_metric.get_fuzzy_luka_order(input_image, patch_size=patch_size, stride=stride, order=2)
        temp = homog_metric.get_glcm_patch_object(input_image, patch_size=patch_size, stride=stride)
        contrast_7T = homog_metric.get_glcm_features(temp, feature_keys=['contrast'])
        temp_dict = {'hi_integral': hi_input,
                     'ssim': ssim_target_input,
                     'fuzzy_luka_2': luka_two,
                     'glcm_contrast': contrast_7T['contrast']}
        metric_values.append(temp_dict)
    metric_dict = hmisc.listdict2dictlist(metric_values)
    return metric_dict


results = {}
for ii, i_file in enumerate(file_list):
    print(ii, end='\r')
    result_dict = run_stuff(ii)
    results[i_file] = result_dict

json_ser_obj = json.dumps(results)
with open('/local_scratch/sharreve/varying_signal_feature_dict_train_data.json', 'w') as f:
    f.write(json_ser_obj)


# Now try to visualize it
with open('/home/bugger/Documents/paper/homogeneity_index/varying_signal_results_hi_ssim_train_data.json', 'r') as f:
    temp = f.read()

varying_signal_train = json.loads(temp)

fig, ax = plt.subplots()
avg_corr = []
for k, v in varying_signal_train.items():
    xvar = v['ssim']
    yvar = v['hi_integral']
    corr, _ = pearsonr(xvar, yvar)
    avg_corr.append(corr)
    if corr >= 0.9:
        ax.scatter(xvar, yvar)

plt.hist(avg_corr, bins=32)