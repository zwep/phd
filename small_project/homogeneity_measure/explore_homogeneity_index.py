#
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
Looks nice..

However, capturing the power series into one index is still tough...
We tried the L2 norm now between the initial HI index and the modified HI index.
(Code for this is also deleted)

We also tried to fit a curve to the HI curve.. to see if we can determine it with a certain parameter
However, an exponential model/log model did no go well.. so we'll leave that for now.
(Code is also deleted)

Currently we are checking if the HI corrleates iwth SSIM somehow.

"""


"""
Load Rho data
"""
ddata_mrl = '/home/bugger/Documents/data/1.5T/prostate/4_MR/MRL/20201228_0004.h5'

with h5py.File(ddata_mrl, 'r') as f:
    temp = f['data']
    n_slice = temp.shape[0]
    A_rho = np.array(f['data'][n_slice//2])


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


import importlib
importlib.reload(data_generator)
data_obj = data_generator.DummyVaryingSignalData(rho=A_rho, b1p=A_b1p, b1m=A_b1m, mask=A_mask, min_degree=1, max_degree=180)
varying_signal_image, varying_bias_field = data_obj.create_varying_signal_maps()


varying_signal_image_cropped, mask_cropped = zip(*[harray.get_crop(x, A_mask) for x in varying_signal_image])
varying_bias_field_cropped, _ = zip(*[harray.get_crop(x, A_mask) for x in varying_bias_field])

hplotc.SlidingPlot(np.array(varying_signal_image_cropped))
hplotc.SlidingPlot(varying_bias_field)

"""
Get RMSE values compared to the homogeenous image
"""
# Scale mask
old_shape = A_mask.shape
n_pixel = 20
new_shape = np.array(A_mask.shape)-n_pixel
import skimage.transform as sktransf
A_mask_resize = sktransf.resize(A_mask, output_shape=new_shape, preserve_range=True)
A_mask_pad = np.pad(A_mask_resize, [(n_pixel//2, n_pixel//2), (n_pixel//2, n_pixel//2)])
hplotc.ListPlot([A_mask, A_mask_pad, A_mask- A_mask_pad])

sel_mask = A_mask_pad
A_rho = np.ma.masked_where(sel_mask == 0, A_rho)
masked_varying_signal_image = np.ma.masked_array([np.ma.masked_where(sel_mask == 0, x) for x in varying_signal_image])
rmse_value = [np.sqrt(np.mean((A_rho - x)**2)) for x in masked_varying_signal_image]
from skimage.metrics import structural_similarity
from scipy.stats import wasserstein_distance
ssim_value = [structural_similarity(A_rho, x) for x in masked_varying_signal_image]
wd_value = [hmisc.patch_min_fun(1, A_rho, x) for x in masked_varying_signal_image]

plt.scatter(rmse_value, ssim_value)
plt.scatter(rmse_value, wd_value)
plt.scatter(ssim_value, wd_value)

from objective.inhomog_removal import CalculateMetrics
calc_metric_obj = CalculateMetrics.CalculateMetrics('/home', '/home', glcm_dist=[1,2,3,4,5])
container = [calc_metric_obj.get_glcm_slice(A_rho, x, patch_size=64) for x in masked_varying_signal_image]
feature_dict_rel, feature_dict_a, feature_dict_b = zip(*container)

homog_rel = [x['homogeneity'] for x in feature_dict_rel]
homog_rel_2 = [(x['homogeneity'] - y['homogeneity']) / y['homogeneity'] for x, y in zip(feature_dict_a, feature_dict_b)]
contr_rel = [(x['contrast'] - y['contrast']) / y['contrast'] for x, y in zip(feature_dict_a, feature_dict_b)]
nrg_rel = [x['energy'] for x in feature_dict_rel]
diss_rel = [(x['dissimilarity'] - y['dissimilarity']) / y['dissimilarity'] for x, y in zip(feature_dict_a, feature_dict_b)]
# diss_rel = [(x['dissimilarity'] - y['dissimilarity']) / y['dissimilarity'] for x, y in zip(feature_dict_a, feature_dict_b)]
corr_rel = [x['correlation'] for x in feature_dict_rel]

def plot_and_correlate(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title(f'Correlation {np.round(np.corrcoef(x, y)[0, 1], 2)}')
    return fig

plot_and_correlate(homog_rel_2[2:], rmse_value[2:])
plot_and_correlate(homog_rel_2[2:], ssim_value[2:])
plot_and_correlate(homog_rel_2[2:], wd_value[2:])

plot_and_correlate(contr_rel[2:], rmse_value[2:])
plot_and_correlate(contr_rel[2:], ssim_value[2:])
plot_and_correlate(contr_rel[2:], wd_value[2:])

# For now.. Energy is the only positive related quantity...
plot_and_correlate(nrg_rel[2:], rmse_value[2:])
# Deze twee hieronder is inderdaad echt het beste...
plot_and_correlate(nrg_rel[4:], ssim_value[4:])
plot_and_correlate(nrg_rel[4:], wd_value[4:])

plot_and_correlate(diss_rel[2:], rmse_value[2:])
plot_and_correlate(diss_rel[2:], ssim_value[2:])
plot_and_correlate(diss_rel[4:], wd_value[4:])

plot_and_correlate(corr_rel, rmse_value)
plot_and_correlate(corr_rel, ssim_value)
plot_and_correlate(corr_rel, wd_value)

"""
Lets test the GLCM features on both the varying signal as well as on the varying bias field
"""

import small_project.homogeneity_measure.metric_implementations as homog_metric

homog_img = []
homog_biasf = []
for x, y in zip(varying_signal_image_cropped, varying_bias_field_cropped):
    x_patches = homog_metric.get_glcm_patch_object(x, patch_size=min(x.shape)//3)
    y_patches = homog_metric.get_glcm_patch_object(y, patch_size=min(x.shape)//3)
    x_glcm = homog_metric.get_glcm_features(x_patches)
    y_glcm = homog_metric.get_glcm_features(y_patches)
    homog_img.append(x_glcm['homogeneity'])
    homog_biasf.append(y_glcm['homogeneity'])

plt.plot(homog_img, 'r')
plt.plot(homog_biasf, 'b')


"""

HI value..
"""

# mask_obj = hplotc.MaskCreator(A_rho * A_mask)
# selected_mask = mask_obj.mask
selected_mask = A_mask
# selected_mask = data_obj.center_mask

metric_values = []
for ii in range(len(varying_signal_image)):
    input_image = varying_signal_image[ii] * selected_mask
    ssim_target_input = skimage.metrics.structural_similarity(input_image, A_rho)
    hi_input = homog_metric.get_hi_value_integral(input_image, selected_mask)
    temp_dict = {'hi_integral': hi_input,
                 'ssim_target_input': ssim_target_input}
    metric_values.append(temp_dict)

# Plot the various metrics
metric_dict = hmisc.listdict2dictlist(metric_values)

fig, ax = plt.subplots()
for k, v in metric_dict.items():
    if k != 'ssim_target_input':
        ax.plot(data_obj.selected_flip_angles_degree, v, label=k)
        ax.twinx().plot(data_obj.selected_flip_angles_degree, metric_dict['ssim_target_input'], c='k',
                                 label='ssim_target_input')
        ax.legend()

fig.savefig('/home/bugger/Documents/paper/homogeneity_index/hi_feature_overview.png')

"""
This concludes that SSIM is VERY dependent on the mask taht is being used

When using a full body mask we get very fluctuating results
When using a mask that focussed on the prostate_mri_mrl (and avoids any noisy areas) we 
get results that dont vary that much
"""

# Now... what can we say about the homogeneity index on the test split...

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

# Quickly create all the masks.. will save some time
# ddata = '/media/bugger/MyBook/data/7T_data/prostate_semireal_data/test_split_results/mask_temp_folder'
# for ii in range(len(target_examples)):
#     print(f'Treating example {ii}', end='\r')
#     file_name = os.path.basename(target_examples[ii])
#     target_file = os.path.join(ddata, file_name)
#     target_array = np.load(target_examples[ii])
#     target_array = harray.scale_minmax(target_array)
#     mask_array = harray.get_treshold_label_mask(target_array)
#     np.save(target_file, mask_array)
# #

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

    metric_values.append(feature_dict)

# Plot the various metrics
metric_dict = hmisc.listdict2dictlist(metric_values)

# Store this stuff
ddest = '/home/bugger/Documents/paper/homogeneity_index/metric_dict_hi.json'
serialized_json = json.dumps(metric_dict)
with open(ddest, 'w') as f:
    f.write(serialized_json)
