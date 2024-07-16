"""
We have a dataset from Bart containing.....


"""

"""
Apply the 1 to 1 model on Bart's data for a 7T/3T comparisson

Vet kut ge-code. Maar het werkt..
"""


import helper.plot_class as hplotc
from skimage.util import img_as_ubyte, img_as_int, img_as_uint
import re
import pydicom
import skimage.transform as sktransform
import os
import helper.misc as hmisc
import SimpleITK as sitk
import helper.array_transf as harray
import objective.inhomog_removal.executor_inhomog_removal as executor
import os
import json
import torch
import numpy as np
import helper.metric as hmetric

dtarget = 'home/bugger/Documents/paper/inhomogeneity removal/compare_many_T'
ddata = '/media/bugger/MyBook/data/multiT_scan/prostaat/prediction_model'

metric_rho_3T_list = []
metric_biasf_3T_list = []
metric_7T_3T_list = []
metric_n4itk_7T_3T_list = []
metric_n4itk_3T_3T_list = []

for i_patient in os.listdir(ddata):
    print(i_patient)
    patient_dir = os.path.join(ddata, i_patient)
    corrected_biasf = os.path.join(patient_dir, 'corrected_biasfield.dcm')
    corrected_rho = os.path.join(patient_dir, 'corrected_rho.dcm')
    uncorrected_3T = os.path.join(patient_dir, 'uncorrected_3T.dcm')
    uncorrected_7T = os.path.join(patient_dir, 'uncorrected_7T.dcm')

    array_7T = pydicom.read_file(uncorrected_7T).pixel_array
    array_3T = pydicom.read_file(uncorrected_3T).pixel_array
    array_rho = pydicom.read_file(corrected_rho).pixel_array
    array_biasf = pydicom.read_file(corrected_biasf).pixel_array
    average_shape = (np.array(array_7T.shape) + np.array(array_3T.shape)) // 2

    array_7T = sktransform.resize(array_7T, average_shape)
    array_3T = sktransform.resize(array_3T, average_shape)
    array_rho = sktransform.resize(array_rho, average_shape)
    array_biasf = sktransform.resize(array_biasf, average_shape)

    mask_3T = harray.get_treshold_label_mask(array_3T)
    mask_7T = harray.get_treshold_label_mask(array_rho)

    array_rho, _ = harray.get_center_transformation(array_rho, mask_7T)
    array_biasf, _ = harray.get_center_transformation(array_biasf, mask_7T)
    array_7T, _ = harray.get_center_transformation(array_7T, mask_7T)
    array_3T, _ = harray.get_center_transformation(array_3T, mask_3T)
    print('\t\tRunning N4ITK...')
    import biasfield_algorithms.N4ITK
    n4itk_7T, n4itk_biasf_7T = biasfield_algorithms.N4ITK.get_n4itk(array_7T, mask=mask_7T)
    n4itk_7T = harray.scale_minmax(n4itk_7T)
    n4itk_3T, n4itk_biasf_3T = biasfield_algorithms.N4ITK.get_n4itk(array_3T, mask=mask_3T)
    n4itk_3T = harray.scale_minmax(n4itk_3T)
    print('\t\tRunning Metrics...')
    metric_rho_3T = hmetric.get_metrics_target(pred=array_rho, target=array_7T)
    metric_biasf_3T = hmetric.get_metrics_target(pred=array_biasf, target=array_3T)
    metric_7T_3T = hmetric.get_metrics_target(pred=array_7T, target=array_3T)
    metric_n4itk_7T_3T = hmetric.get_metrics_target(pred=n4itk_7T, target=array_3T)
    metric_n4itk_3T_3T = hmetric.get_metrics_target(pred=n4itk_3T, target=array_3T)
    print('\t\tRunning Distr Metrics...')
    metric_distr_rho_3T = hmetric.get_metrics_distribution_target(pred=array_rho, target=array_7T, mask_pred=mask_7T, mask_target=mask_3T)
    metric_distr_biasf_3T = hmetric.get_metrics_distribution_target(pred=array_biasf, target=array_3T, mask_pred=mask_7T, mask_target=mask_3T)
    metric_distr_7T_3T = hmetric.get_metrics_distribution_target(pred=array_7T, target=array_3T, mask_pred=mask_7T, mask_target=mask_3T)
    metric_distr_n4itk_7T_3T = hmetric.get_metrics_distribution_target(pred=n4itk_7T, target=array_3T, mask_pred=mask_7T, mask_target=mask_3T)
    metric_distr_n4itk_3T_3T = hmetric.get_metrics_distribution_target(pred=n4itk_3T, target=array_3T, mask_pred=mask_7T, mask_target=mask_3T)

    metric_rho_3T.update(metric_distr_rho_3T)
    metric_biasf_3T.update(metric_distr_biasf_3T)
    metric_7T_3T.update(metric_distr_7T_3T)
    metric_n4itk_7T_3T.update(metric_distr_n4itk_7T_3T)
    metric_n4itk_3T_3T.update(metric_distr_n4itk_3T_3T)

    metric_rho_3T_list.append(metric_rho_3T)
    metric_biasf_3T_list.append(metric_biasf_3T)
    metric_7T_3T_list.append(metric_7T_3T)
    metric_n4itk_7T_3T_list.append(metric_n4itk_7T_3T)
    metric_n4itk_3T_3T_list.append(metric_n4itk_3T_3T)


print('Done predicting')
import pandas as pd
data_frame_rho_3T = pd.DataFrame.from_dict(metric_rho_3T_list)
data_frame_rho_3T.to_csv(os.path.join(dtarget, 'rho_3T.csv'), index=False)

data_frame_biasf_3T = pd.DataFrame.from_dict(metric_biasf_3T_list)
data_frame_biasf_3T.to_csv(os.path.join(dtarget, 'biasf_3T.csv'), index=False)

data_frame_7T_3T = pd.DataFrame.from_dict(metric_7T_3T_list)
data_frame_7T_3T.to_csv(os.path.join(dtarget, '7T_3T.csv'), index=False)

data_frame_n4itk_7T_3T = pd.DataFrame.from_dict(metric_n4itk_7T_3T_list)
data_frame_n4itk_7T_3T.to_csv(os.path.join(dtarget, 'n4itk_7T_3T.csv'), index=False)

data_frame_n4itk_3T_3T = pd.DataFrame.from_dict(metric_n4itk_3T_3T_list)
data_frame_n4itk_3T_3T.to_csv(os.path.join(dtarget, 'n4itk_3T_3T.csv'), index=False)

print('Done')