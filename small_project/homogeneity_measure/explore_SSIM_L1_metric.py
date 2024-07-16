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
from skimage.metrics import structural_similarity
ssim_image = [[structural_similarity(A_rho, x).round(2)] for x in varying_signal_image]
L1_image = [[np.abs(A_rho - x).mean().round(4)] for x in varying_signal_image]

plot_array = np.concatenate([(A_rho * A_mask)[None], varying_signal_image[::5]], axis=0)
hplotc.ListPlot(plot_array, col_row=(3, 3), ax_off=True, subtitle=[[1], *ssim_image[::5]])
hplotc.ListPlot(plot_array, col_row=(3, 3), ax_off=True, subtitle=[[0], *L1_image[::5]])
