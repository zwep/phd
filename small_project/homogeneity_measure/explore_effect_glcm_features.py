from scipy.stats import pearsonr
import json
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

"""
The GLCM can offer some features about texture analysis

Homogeneity is one of them.. Lets see if we can reflect the appeared inhomogeneity increase

Als het goed is laat dit zien dat features van de GLCM NIET handig zijn... 
"""

"""
Load Rho data (not registered to a specific b1+ map....)
"""

ddata_mrl = '/home/bugger/Documents/data/1.5T/prostate_mri_mrl/4_MR/MRL/20201228_0004.h5'

with h5py.File(ddata_mrl, 'r') as f:
    temp = f['data']
    n_slice = temp.shape[0]
    A_rho = np.array(f['data'][n_slice // 2])

A_rho = skimage.transform.resize(A_rho, (256, 256), preserve_range=True, anti_aliasing=False)
A_rho = harray.scale_minmax(A_rho)

"""
Load B1 data
"""

flavio_data = '/home/bugger/Documents/data/test_clinic_registration/flavio_data/M01.mat'

A = scipy.io.loadmat(flavio_data)
A_b1p = np.moveaxis(A['Model']['B1plus'][0][0], -1, 0)
A_b1m = np.moveaxis(A['Model']['B1minus'][0][0], -1, 0)
A_mask = A['Model']['Mask'][0][0]

data_obj = data_generator.DummyVaryingSignalData(rho=A_rho, b1p=A_b1p, b1m=A_b1m, mask=A_mask)
varying_signal_image = data_obj.create_varying_signal_maps()


"""
Now measure the SSIM and the GLCM version
"""

metric_values = []
for ii in range(len(varying_signal_image)):
    input_image = varying_signal_image[ii] * A_mask
    ssim_target_input = skimage.metrics.structural_similarity(input_image, A_rho)
    glcm_obj_input = homog_metric.get_glcm_patch_object(input_image)
    glcm_features_input = homog_metric.get_glcm_features(glcm_obj_input)
    feature_dict = {'ssim_target_input': ssim_target_input}
    feature_dict.update(glcm_features_input)
    metric_values.append(feature_dict)

# Plot the various metrics
metric_dict = hmisc.listdict2dictlist(metric_values)

print('Number of features to be plotted...', len(metric_dict))
fig = plt.figure(figsize=(15, 10))
ax = fig.subplots(2, 4)
ax = ax.ravel()
counter = 0
for k, v in metric_dict.items():
    if k != 'ssim_target_input':
        ax[counter].plot(data_obj.selected_flip_angles_degree, v, label=k)
        ax[counter].twinx().plot(data_obj.selected_flip_angles_degree, metric_dict['ssim_target_input'], c='k',
                                 label='ssim_target_input')
        ax[counter].legend()
        counter += 1

fig.savefig('/home/bugger/Documents/paper/homogeneity_index/glcm_feature_overview.png')


"""
This first test shows little promise though...

However, we want to have a complete comparisson
"""

# Now load this local stuff example...
ddata = '/media/bugger/MyBook/data/7T_data/prostate_semireal_data/test_split_results/single_homogeneous'
ddata_mask = '/media/bugger/MyBook/data/7T_data/prostate_semireal_data/test_split_results/mask_temp_folder'

temp_input_examples = [x[len('input'):] for x in os.listdir(ddata) if 'input' in x]
temp_target_examples = [x[len('target'):] for x in os.listdir(ddata) if 'target' in x]
temp_pred_examples = [x[len('pred'):] for x in os.listdir(ddata) if 'pred' in x]
temp_mask_examples = [x[len('target'):] for x in os.listdir(ddata_mask) if 'target' in x]

intersection_files = set(temp_target_examples).intersection(set(temp_input_examples))
input_examples = sorted([os.path.join(ddata, 'input' + x) for x in temp_input_examples])
target_examples = sorted([os.path.join(ddata, 'target' + x) for x in temp_target_examples])
pred_examples = sorted([os.path.join(ddata, 'pred' + x) for x in temp_pred_examples])
mask_examples = sorted([os.path.join(ddata_mask, 'target' + x) for x in temp_mask_examples])


metric_values = []
for ii in range(len(target_examples)):
# for ii in range(3):
    print(f'Treating example {ii} - {os.path.basename(target_examples[ii])}             ', end='\r')
    # Images are around 528 - 640 in size
    # The images above are 256... so a patch of 32 seems OK
    target_array = np.load(target_examples[ii])
    target_array = harray.scale_minmax(target_array)
    mask_array = np.load(mask_examples[ii])
    input_array = np.load(input_examples[ii])
    input_array = harray.scale_minmax(input_array)
    pred_array = np.load(pred_examples[ii])
    pred_array = harray.scale_minmax(pred_array)

    glcm_input_obj = homog_metric.get_glcm_patch_object(input_array, patch_size=64)
    glcm_input_features = homog_metric.get_glcm_features(glcm_input_obj, key_appendix='_input')

    glcm_target_obj = homog_metric.get_glcm_patch_object(target_array, patch_size=64)
    glcm_target_features = homog_metric.get_glcm_features(glcm_target_obj, key_appendix='_target')

    glcm_pred_obj = homog_metric.get_glcm_patch_object(pred_array, patch_size=64)
    glcm_pred_features = homog_metric.get_glcm_features(glcm_pred_obj, key_appendix='_pred')

    ## Standaard metrics - SSIM
    ssim_target_input = skimage.metrics.structural_similarity(input_array, target_array)
    ssim_target_pred = skimage.metrics.structural_similarity(pred_array, target_array)

    ## Comparisson with
    hi_integral_input = homog_metric.get_hi_value_integral(input_array, mask_array)
    hi_integral_target = homog_metric.get_hi_value_integral(target_array, mask_array)
    hi_integral_pred = homog_metric.get_hi_value_integral(pred_array, mask_array)

    feature_dict = {'hi_integral_input': hi_integral_input,
                    'hi_integral_target': hi_integral_target,
                    'hi_integral_pred': hi_integral_pred,
                    'ssim_target_input': ssim_target_input,
                    'ssim_target_pred': ssim_target_pred}
    feature_dict.update(glcm_input_features)
    feature_dict.update(glcm_target_features)
    feature_dict.update(glcm_pred_features)
    metric_values.append(feature_dict)


metric_listdict = hmisc.listdict2dictlist(metric_values)

# I should have these already available...
ddest = '/home/bugger/Documents/paper/homogeneity_index/metric_dict_glcm.json'
serialized_json = json.dumps(metric_listdict)
with open(ddest, 'w') as f:
    f.write(serialized_json)
