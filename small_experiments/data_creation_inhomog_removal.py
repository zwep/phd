
import data_generator.InhomogRemoval as data_gen
import helper.array_transf
import helper.plot_class as hplotc
import numpy as np
import os

import os
import nrrd
import nrrd.errors
import re
import torch
import matplotlib.pyplot as plt
import pandas as pd
import helper.array_transf as harray
from PIL import Image
import skimage.transform as sktransform
import helper.misc as hmisc
import tooling.shimming.b1shimming_single as mb1_single
import os
import collections
import h5py
import scipy.optimize

import small_project.sinp3.signal_equation as signal_eq

"""
I dont trust the way the data is created...
Here we create some code that can be run remotely to store some input examples...
"""



def scale_certain_angle(x, mask, flip_angle=np.pi / 2):
    # Use a (binary) mask to determine the average signal
    x_sub = x * mask
    # Mean.. of max..?
    x_mean = np.abs(x_sub.sum()) / np.sum(mask)
    # Taking the absolute values to make sure that values are between 0..1
    # B1 plus interference by complex sum. Then using abs value to scale
    # Add some randomness to the flipangle....
    # flip_angle = np.random.uniform(flip_angle - np.pi / 18, flip_angle + np.pi / 18)
    flip_angle_map = np.abs(x) / x_mean * flip_angle
    return flip_angle_map


def improved_scale_signal_model(x):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3310288/
    # Musculoskeletal MRI at 3.0T and 7.0T: A Comparison of Relaxation Times and Image Contrast
    TR_se = 5000
    TE_se = 53
    T1_fat = 583
    T2_fat = 46
    T1_muscle = 1552
    T2_muscle = 23
    T1 = (T1_fat + T1_muscle) / 2
    T2 = (T2_fat + T2_muscle) / 2
    general_signal_se = signal_eq.get_t2_signal_general(flip_angle=x, T1=T1, TE=TE_se, TR=TR_se, T2=T2, N=1,
                                                        beta=flip_angle * 2)
    return general_signal_se

# img_store_dir = '/local_scratch/sharreve/display_image_generation'
# dir_data = '/local_scratch/sharreve/mri_data/registrated_h5/train'
img_store_dir = None
dir_data = '/home/bugger/Documents/data/test_clinic_registration/registrated_h5/test'
input_dir = os.path.join(dir_data, 'input')
target_dir = os.path.join(dir_data, 'target')
target_clean_dir = target_dir + "_clean"
mask_dir = os.path.join(dir_data, 'mask')
file_list = os.listdir(input_dir)
index = 0
relative_phase = True
objective_shim = 'b1'
flip_angle = np.pi / 2
debug = True
i_file = file_list[index]

b1_minus_file = os.path.join(input_dir, i_file)
b1_plus_file = os.path.join(target_dir, i_file)
mask_file = os.path.join(mask_dir, i_file)
target_clean = os.path.join(target_clean_dir, i_file)

with h5py.File(target_clean, 'r') as h5_obj:
    max_slice = h5_obj['data'].shape[0]

sel_slice = np.random.randint(max_slice)

with h5py.File(target_clean, 'r') as f:
    rho_array = np.array(f['data'][sel_slice])

print('minmaxmean', harray.get_minmeanmediammax(rho_array))
rho_array = harray.scale_minpercentile_both(rho_array, q=99, is_complex=False)
# Plot the rho array max
plot_obj = hplotc.ListPlot(rho_array, cbar=True)
if img_store_dir is not None:
    plot_obj.figure.savefig(os.path.join(img_store_dir, 'rho_array_scaled.png'))

if os.path.isfile(mask_file):
    with h5py.File(mask_file, 'r') as f:
        mask_array = np.array(f['data'][sel_slice])
else:
    mask_array = harray.get_treshold_label_mask(rho_array)

import scipy.ndimage
mask_array = scipy.ndimage.binary_fill_holes(harray.smooth_image(mask_array, n_kernel=8))

with h5py.File(b1_plus_file, 'r') as f:
    b1_plus_array = np.array(f['data'][sel_slice])

with h5py.File(b1_minus_file, 'r') as f:
    b1_minus_array = np.array(f['data'][sel_slice])

b1_minus_array = harray.correct_mask_value(b1_minus_array, mask_array)
b1_plus_array = harray.correct_mask_value(b1_plus_array, mask_array)

n_c, n_y, n_x = b1_plus_array.shape
y_offset = np.random.randint(-n_y // 8, n_y // 8)
x_offset = np.random.randint(-n_y // 8, n_y // 8)
y_center = n_y // 2 + y_offset
x_center = n_x // 2 + x_offset
shim_mask = np.zeros((n_y, n_x))
delta_x = int(0.1 * n_y)
shim_mask[y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x] = 1

# Display shim mask..
plot_obj = hplotc.ListPlot(rho_array * (1 + shim_mask))
if img_store_dir is not None:
    plot_obj.figure.savefig(os.path.join(img_store_dir, 'shim_mask.png'))

# Display b1 array
plot_obj = hplotc.ListPlot(b1_plus_array, augm='np.angle')
if img_store_dir is not None:
    plot_obj.figure.savefig(os.path.join(img_store_dir, 'initial_b1p_array_angle.png'))

if relative_phase:
    b1_plus_array = b1_plus_array * np.exp(-1j * np.angle(b1_plus_array[0]))
    b1_plus_array = harray.correct_mask_value(b1_plus_array, mask_array)

plot_obj = hplotc.ListPlot(b1_plus_array, augm='np.angle')
if img_store_dir is not None:
    plot_obj.figure.savefig(os.path.join(img_store_dir, 'relative_b1p_array_angle.png'))

shimming_obj = mb1_single.ShimmingProcedure(b1_plus_array, shim_mask,
                                            relative_phase=relative_phase,
                                            str_objective=objective_shim,
                                            debug=debug)

x_opt, final_value = shimming_obj.find_optimum()
b1_plus_array_shimmed = harray.apply_shim(b1_plus_array, cpx_shim=x_opt)
# Shimmed B1 plus
plot_obj = hplotc.ListPlot(b1_plus_array_shimmed, augm='np.abs', cbar=True)
if img_store_dir is not None:
    plot_obj.figure.savefig(os.path.join(img_store_dir, 'shim_b1p_array.png'))

x_sub = b1_plus_array_shimmed * shim_mask
x_mean = np.abs(x_sub.mean())
print('MEAN VALUE OF X_MEAN', x_mean)

b1_plus_array_shimmed = harray.scale_minpercentile_both(b1_plus_array_shimmed, q=99, is_complex=False)

x_sub = b1_plus_array_shimmed * shim_mask
x_mean = np.abs(x_sub.mean())
print('MEAN VALUE OF X_MEAN', x_mean)

flip_angle_map = scale_certain_angle(b1_plus_array_shimmed, flip_angle=flip_angle, mask=shim_mask)
# Shimmed B1 plus
plot_obj = hplotc.ListPlot(flip_angle_map, augm='np.real', cbar=True)
if img_store_dir is not None:
    plot_obj.figure.savefig(os.path.join(img_store_dir, 'flip_angle_b1p_array.png'))

fig, ax = plt.subplots(3)
n = flip_angle_map.shape[0]
ax[0].plot(flip_angle_map[n // 2])
ax[1].plot(np.abs(b1_plus_array_shimmed[n // 2]))
ax[2].plot(np.abs(b1_plus_array_shimmed[n // 2] * shim_mask[y_center]))
if img_store_dir is not None:
    fig.savefig(os.path.join(img_store_dir, 'flip_angle_b1p_array_line.png'))

b1p_signal_model = improved_scale_signal_model(flip_angle_map)
plot_obj = hplotc.ListPlot(b1p_signal_model, augm='np.real', cbar=True)
if img_store_dir is not None:
    plot_obj.figure.savefig(os.path.join(img_store_dir, 'signal_model_b1p_array.png'))

b1p_signal_model = harray.scale_minpercentile_both(b1p_signal_model, q=99, is_complex=False)
b1_minus_array = harray.scale_minpercentile_both(b1_minus_array, q=99, is_complex=True)

bias_field_array = np.abs(b1_minus_array).sum(axis=0) * b1p_signal_model
bias_field_array = harray.scale_minmax(bias_field_array)
plot_obj = hplotc.ListPlot(bias_field_array, augm='np.real', cbar=True)
if img_store_dir is not None:
    plot_obj.figure.savefig(os.path.join(img_store_dir, 'biasfield_array.png'))

input_array = rho_array * (b1_minus_array) * (b1p_signal_model)
input_array = harray.correct_mask_value(input_array, mask_array)
plot_obj = hplotc.ListPlot(input_array, augm='np.real', cbar=True)
if img_store_dir is not None:
    plot_obj.figure.savefig(os.path.join(img_store_dir, 'input_array.png'))

input_array = harray.scale_minpercentile_both(input_array, q=99, is_complex=True)
# ind_trunc = np.abs(input_array) > 1
# input_array[ind_trunc] = input_array[ind_trunc] / np.abs(input_array[ind_trunc])
plot_obj = hplotc.ListPlot(input_array, augm='np.real', cbar=True)
if img_store_dir is not None:
    plot_obj.figure.savefig(os.path.join(img_store_dir, 'input_array_trunc.png'))

target_array = bias_field_array
target_array = harray.scale_minpercentile_both(target_array, q=99, is_complex=True)
# ind_trunc = np.abs(target_array) > 1
# target_array[ind_trunc] = target_array[ind_trunc] / np.abs(target_array[ind_trunc])

plot_obj = hplotc.ListPlot(target_array, augm='np.real', cbar=True)
if img_store_dir is not None:
    plot_obj.figure.savefig(os.path.join(img_store_dir, 'target_array_trunc.png'))


"""
Check the different preprocessing steps to create the input/target for the bias field model...

I want to avoid dividing by zero.
"""
# Now move on with the Bias field calculations... check if we can recover the input...
# And see how this goes when we standardize the images... (mean/std)

input_array = rho_array * (b1_minus_array) * (b1p_signal_model)
input_array = harray.correct_mask_value(input_array, mask_array)
input_array = harray.scale_minpercentile_both(input_array, q=99, is_complex=True)
input_array = harray.scale_minmax(input_array, is_complex=True)

bias_field_array = np.abs(b1_minus_array).sum(axis=0) * b1p_signal_model
bias_field_array = harray.scale_minmax(bias_field_array, is_complex=True)

cor_image = np.abs(input_array).mean(axis=0)/bias_field_array
cor_image = helper.array_transf.correct_inf_nan(cor_image)
hplotc.ListPlot([np.abs(input_array).mean(axis=0), cor_image, bias_field_array], cbar=True)

multi_coil_biasfield = (b1_minus_array) * (b1p_signal_model)
# multi_coil_biasfield = 1 + 1j + (multi_coil_biasfield - multi_coil_biasfield.mean(axis=(-2, -1), keepdims=True)) / np.abs(multi_coil_biasfield).std(axis=(-2, -1), keepdims=True)

bias_field_array = np.abs(np.abs(b1_minus_array).mean(axis=0) * b1p_signal_model)
# bias_field_array = harray.scale_minmax(bias_field_array)
bias_field_array = 1 + (bias_field_array - bias_field_array.mean(axis=(-2, -1), keepdims=True)) / np.abs(bias_field_array).std(axis=(-2, -1), keepdims=True)

input_array = rho_array * multi_coil_biasfield
input_array = harray.scale_minpercentile_both(input_array, q=99, is_complex=True)

hplotc.ListPlot([np.abs(input_array).sum(axis=0), np.abs(input_array).sum(axis=0)/bias_field_array, bias_field_array])