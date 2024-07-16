from scipy.stats import pearsonr
import json
import os
import matplotlib.pyplot as plt
import helper.array_transf as harray
import h5py
import helper.misc as hmisc
import numpy as np
import scipy.io
import skimage.transform
import skimage.metrics
import scipy.integrate
import small_project.homogeneity_measure.create_dummy_data as data_generator
import small_project.homogeneity_measure.metric_implementations as homog_metric

"""
Ander paper

New measures of homogeneity for imageprocessing: an application to fignerprinting segmetnaiton
 
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
Now start calculating stuff on the varying intensity data

Just to see if any related to the SSIM
"""

import skimage.feature

feature_images = []
patch_size = 64
stride = patch_size // 2
for temp_img in varying_signal_image:
    feature_dict = homog_metric.get_fuzzy_features(temp_img)
    feature_images.append(feature_dict)

metric_dict = hmisc.listdict2dictlist(feature_images)


print('Number of features to be plotted...', len(metric_dict))
fig = plt.figure(figsize=(15, 10))
ax = fig.subplots(2, 3)
ax = ax.ravel()
counter = 0
for k, v in metric_dict.items():
    if k != 'ssim_target_input':
        ax[counter].plot(data_obj.selected_flip_angles_degree, v, label=k)
        ax[counter].twinx().plot(data_obj.selected_flip_angles_degree, metric_dict['ssim_target_input'], c='k',
                                 label='ssim_target_input')
        ax[counter].legend()
        counter += 1

fig.savefig('/home/bugger/Documents/paper/homogeneity_index/fuzzy_feature_overview.png')


"""
There is one that looks OK!
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
    print(f'Treating example {ii} - {os.path.basename(target_examples[ii])}             ', end='\r')
    target_array = np.load(target_examples[ii])
    target_array = harray.scale_minmax(target_array)
    mask_array = np.load(mask_examples[ii])
    input_array = np.load(input_examples[ii])
    input_array = harray.scale_minmax(input_array)
    pred_array = np.load(pred_examples[ii])
    pred_array = harray.scale_minmax(pred_array)

    ## Fuzzy metrics - input
    fuzzy_features_input = homog_metric.get_fuzzy_features(input_array, key_appendix='_input')
    fuzzy_features_pred = homog_metric.get_fuzzy_features(pred_array, key_appendix='_pred')
    fuzzy_features_target = homog_metric.get_fuzzy_features(target_array, key_appendix='_target')

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

    feature_dict.update(fuzzy_features_input)
    feature_dict.update(fuzzy_features_target)
    feature_dict.update(fuzzy_features_pred)
    metric_values.append(feature_dict)


metric_listdict = hmisc.listdict2dictlist(metric_values)
# I should have these already available...
ddest = '/home/bugger/Documents/paper/homogeneity_index/metric_dict_fuzzy.json'
serialized_json = json.dumps(metric_listdict)
with open(ddest, 'w') as f:
    f.write(serialized_json)
