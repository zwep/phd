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
GOD here we go again...
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
import small_project.homogeneity_measure.metric_implementations as homog_metric
importlib.reload(data_generator)
data_obj = data_generator.DummyVaryingSignalData(rho=A_rho, b1p=A_b1p, b1m=A_b1m, mask=A_mask, min_degree=1, max_degree=180)
data_obj.selected_flip_angles_degree = np.arange(1, 180, 1)
varying_signal_map, varying_bias_field = data_obj.create_varying_signal_maps()
masked_varying_signal_map = np.ma.masked_array([np.ma.masked_where(A_mask == 0, x) for x in varying_signal_map])

masked_rho = np.ma.masked_where(A_mask == 0, A_rho)

hplotc.SlidingPlot(masked_varying_signal_map)

patch_size = 128
glcm_dist_max = 10
glcm_dist = list(np.linspace(1, glcm_dist_max, 5))
feature_keys = ['homogeneity']
content_list = []
for x in varying_signal_map:
    glcm_obj_a = homog_metric.get_glcm_patch_object(x, patch_size=patch_size, glcm_dist=glcm_dist)
    feature_dict_a = {}
    n_patches = float(len(glcm_obj_a))
    for patch_obj_a in glcm_obj_a:
        for i_feature in feature_keys:
            _ = feature_dict_a.setdefault(i_feature, 0)
            feature_value_a = skimage.feature.graycoprops(patch_obj_a, i_feature)
            feature_dict_a[i_feature] += np.mean(feature_value_a) / n_patches
    homog_value = feature_dict_a['homogeneity']
    content_dict = {'dist': glcm_dist, 'patch': patch_size, 'homog': homog_value}
    content_list.append(content_dict)


# #
glcm_obj_a = homog_metric.get_glcm_patch_object(masked_rho, patch_size=patch_size, glcm_dist=glcm_dist)
feature_dict_a = {}
n_patches = float(len(glcm_obj_a))
for patch_obj_a in glcm_obj_a:
    for i_feature in feature_keys:
        _ = feature_dict_a.setdefault(i_feature, 0)
        feature_value_a = skimage.feature.graycoprops(patch_obj_a, i_feature)
        feature_dict_a[i_feature] += np.mean(feature_value_a) / n_patches
homog_value = feature_dict_a['homogeneity']
content_dict = {'dist': glcm_dist, 'patch': patch_size, 'homog': homog_value}
# #
plt.plot([x['homog'] for x in content_list])

from skimage.util import img_as_ubyte, img_as_uint
plt.hist(img_as_ubyte(harray.scale_minmax(masked_rho.data[A_mask==1].ravel())), bins=256)
plt.hist(img_as_ubyte(harray.scale_minmax(varying_signal_map[0][A_mask==1].ravel())), bins=256, label='inhomog')
plt.legend()

hplotc.ListPlot([masked_rho, varying_signal_map[0]])
# #
feature_rel_dict, feature_input_dict, feature_target_dict = zip(*[homog_metric.get_relative_glcm_features(x, A_rho) for x in varying_signal_map])

# # #
selected_inhomog_image = varying_signal_map[0]
selected_inhomog_image, _ = harray.get_crop(selected_inhomog_image, A_mask)
selected_inhomog_image = harray.scale_minmax(selected_inhomog_image)
selected_homog_image = A_rho
selected_homog_image, _ = harray.get_crop(selected_homog_image * A_mask, A_mask)
selected_homog_image = harray.scale_minmax(selected_homog_image)
hplotc.ListPlot([selected_homog_image, selected_inhomog_image])
glcm_dist = [5,6,7,8,9]
patch_size = min(selected_inhomog_image.shape) // 4

equil_obj = hplotc.ImageIntensityEqualizer(reference_image=selected_homog_image, image_list=[selected_inhomog_image],
                                           patch_width=50, dynamic_thresholding=True)
A_smooth_unity_equalized = equil_obj.correct_image_list()[0]

# selected_homog_image[selected_homog_image > equil_obj.vmax_ref] = equil_obj.vmax_ref
hplotc.ListPlot([A_smooth_unity_equalized, selected_homog_image])
hplotc.ListPlot([np.array(equil_obj.patches_image_list), equil_obj.patches_ref])

input_patches = harray.get_patches(selected_inhomog_image, patch_shape=(patch_size, patch_size), stride=patch_size)
target_patches = harray.get_patches(selected_homog_image, patch_shape=(patch_size, patch_size), stride=patch_size)
hplotc.ListPlot([input_patches])
hplotc.ListPlot([target_patches])
glcm_obj_input = homog_metric.get_glcm_patch_object(selected_inhomog_image, patch_size=patch_size, glcm_dist=glcm_dist)
glcm_obj_target = homog_metric.get_glcm_patch_object(selected_homog_image, patch_size=patch_size, glcm_dist=glcm_dist)
feature_input_dict = {}
feature_target_dict = {}
n_patches = float(len(glcm_obj_target))
for input_patch_obj, target_patch_obj in zip(glcm_obj_input, glcm_obj_target):
    for i_feature in ['homogeneity']:
        feature_target_dict.setdefault(i_feature, [])
        feature_input_dict.setdefault(i_feature, [])
        target_feature_value = skimage.feature.graycoprops(target_patch_obj, i_feature)
        input_feature_value = skimage.feature.graycoprops(input_patch_obj, i_feature)
        feature_target_dict[i_feature].append(target_feature_value)
        feature_input_dict[i_feature].append(input_feature_value)

feature_input_dict['homogeneity'][0]
plt.plot([x.mean() for x in feature_input_dict['homogeneity']])
plt.plot([x.mean() for x in feature_target_dict['homogeneity']])
# #

fig, ax = plt.subplots(2)
ax[0].plot([x['homogeneity'] for x in feature_input_dict], label='homog input')
ax[0].plot([x['homogeneity'] for x in feature_target_dict], label='homog target')

res_homog = [x['homogeneity'] for x in glcm_metrics]
res_energy = [x['energy'] for x in glcm_metrics]
res_contrast = [x['contrast'] for x in glcm_metrics]

import scipy.spatial
import scipy.stats
from skimage.metrics import structural_similarity
import helper.metric as hmetric
jensen_shannon_distance = [scipy.spatial.distance.jensenshannon(x.ravel(), A_rho.ravel()) for x in varying_signal_map]
chi_squared_distance = [hmetric.chi_squared_distance(x, A_rho) for x in varying_signal_map]
wasserstein_distance = [scipy.stats.wasserstein_distance(x.ravel(), A_rho.ravel()) for x in varying_signal_map]
ssim_dist = [structural_similarity(x, A_rho) for x in varying_signal_map]
res_rmse = [np.sqrt(np.mean((x - A_rho)**2)) for x in varying_signal_map]

import small_project.homogeneity_measure.metric_implementations as derpmetric
res_hi = [derpmetric.get_hi_value_integral(x, A_rho) for x in varying_signal_map]

fig, ax = plt.subplots(5)
ax[0].plot(jensen_shannon_distance, label='jensehshannon')
ax[1].plot(chi_squared_distance, label='chi squared')
ax[2].plot(wasserstein_distance, label='wss')
ax[3].plot(ssim_dist, label='ssim')
ax[4].plot(res_rmse, label='rmse')
for selax in range(5):
    ax[selax].legend()

fig, ax = plt.subplots(4)
ax[0].plot(res_contrast, label='contrast')
ax[1].plot(res_energy, label='energy')
ax[2].plot(res_homog, label='homog')
ax[3].plot(res_hi, label='hi')

# ax.twinx().plot(res_contrast, 'r')
# ax.twinx().plot(res_energy, 'g')
# Waarom is RMSE zo slecht...?