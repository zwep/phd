"""
RMSE shows weird results..

Lets manually check it..
"""

import os
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=6
from multiprocessing import set_start_method
set_start_method("spawn")
import json
import skimage.metrics
import helper.misc as hmisc
import helper.array_transf as harray
import small_project.homogeneity_measure.metric_implementations as homog_metric
import numpy as np
import matplotlib.pyplot as plt
import helper.plot_class as hplotc
import skimage.feature
import re
import objective.inhomog_removal.CalculateMetrics as CalcMetrics
import time
import scipy.stats
import helper.metric as hmetric
base_pred = '/local_scratch/sharreve/model_run/selected_inhomog_removal_models'

"""
Define all paths for volunteer data..
"""

body_mask_dir = '/local_scratch/sharreve/mri_data/volunteer_data/body_mask'
sel_mask_dir = body_mask_dir


single_biasf_volunteer = {"dinput": os.path.join(base_pred, 'single_biasfield/volunteer_corrected/input'),
                          "dpred": os.path.join(base_pred, 'single_biasfield/volunteer_corrected/pred'),
                          "dmask": sel_mask_dir,
                          "name": "volunteer"}

multi_biasf_volunteer = {"dinput": os.path.join(base_pred, 'multi_biasfield/volunteer_corrected/input'),
                         "dpred": os.path.join(base_pred, 'multi_biasfield/volunteer_corrected/pred'),
                         "dmask": sel_mask_dir,
                          "name": "volunteer"}

multi_homog_volunteer = {"dinput": os.path.join(base_pred, 'multi_homogeneous/volunteer_corrected/input'),
                        "dpred": os.path.join(base_pred, 'multi_homogeneous/volunteer_corrected/pred'),
                        "dmask": sel_mask_dir,
                          "name": "volunteer"}

single_homog_volunteer = {"dinput": os.path.join(base_pred, 'single_homogeneous/volunteer_corrected/input'),
                          "dpred": os.path.join(base_pred, 'single_homogeneous/volunteer_corrected/pred'),
                          "dmask": sel_mask_dir,
                          "name": "volunteer"}

n4itk_volunteer = {"dinput": "/local_scratch/sharreve/mri_data/volunteer_data/t2w_n4itk/input",
                   "dpred": "/local_scratch/sharreve/mri_data/volunteer_data/t2w_n4itk/pred",
                   "dmask": sel_mask_dir,
                   "name": "volunteer"}



"""
Loool do this stuff for ... patient data
"""

body_mask_dir = '/local_scratch/sharreve/mri_data/daan_reesink/mask'
sel_mask_dir = body_mask_dir


single_biasf_patient = {"dinput": os.path.join(base_pred, 'single_biasfield/patient_corrected/input'),
                          "dpred": os.path.join(base_pred, 'single_biasfield/patient_corrected/pred'),
                          "dmask": sel_mask_dir,
                          "name": "patient"}

single_homog_patient = {"dinput": os.path.join(base_pred, 'single_homogeneous/patient_corrected/input'),
                          "dpred": os.path.join(base_pred, 'single_homogeneous/patient_corrected/pred'),
                          "dmask": sel_mask_dir,
                          "name": "patient"}

n4itk_patient = {"dinput": "/local_scratch/sharreve/mri_data/daan_reesink/image_n4itk/input",
                    "dpred": "/local_scratch/sharreve/mri_data/daan_reesink/image_n4itk/pred",
                    "dmask": sel_mask_dir,
                          "name": "patient"}

"""
Patient 3T paths 
"""

body_mask_dir = '/local_scratch/sharreve/mri_data/prostate_weighting_h5/test/mask'
sel_mask_dir = body_mask_dir


single_biasf_patient_3T = {"dinput": os.path.join(base_pred, 'single_biasfield/patient_corrected_3T/input'),
                          "dpred": os.path.join(base_pred, 'single_biasfield/patient_corrected_3T/pred'),
                          "dmask": sel_mask_dir,
                          "name": "patient_3T"}

single_homog_patient_3T = {"dinput": os.path.join(base_pred, 'single_homogeneous/patient_corrected_3T/input'),
                          "dpred": os.path.join(base_pred, 'single_homogeneous/patient_corrected_3T/pred'),
                          "dmask": sel_mask_dir,
                          "name": "patient_3T"}

n4itk_patient_3T = {"dinput": "/local_scratch/sharreve/mri_data/prostate_weighting_h5/test/target",
                 "dpred": "/local_scratch/sharreve/mri_data/prostate_weighting_h5/test/target_corrected_N4",
                 "dmask": sel_mask_dir,
                          "name": "patient_3T"}


"""
Patient 1.5T paths 
"""

body_mask_dir = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/mask_b1'
fat_mask_dir = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/mask'
input_n4_test = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/input_abs_sum'
sel_mask_dir = body_mask_dir


single_biasf_test = {"dinput": input_n4_test, #os.path.join(base_pred, 'single_biasfield/target_corrected/input'),
                     "dpred": os.path.join(base_pred, 'single_biasfield/target_corrected/pred'),
                     "dtarget": os.path.join(base_pred, 'single_biasfield/target_corrected/target'),
                     "dmask": sel_mask_dir,
                     "dmask_fat": fat_mask_dir,
                     "name": "test"}

single_homog_test = {"dinput": input_n4_test, #os.path.join(base_pred, 'single_homogeneous/target_corrected/input'),
                     "dpred": os.path.join(base_pred, 'single_homogeneous/target_corrected/pred'),
                     "dtarget": os.path.join(base_pred, 'single_homogeneous/target_corrected/target'),
                     "dmask": sel_mask_dir,
                     "dmask_fat": fat_mask_dir,
                     "name": "test"}

# Using single as input.. to avoid any differences...
multi_biasf_test = {"dinput": os.path.join(base_pred, 'single_biasfield/target_corrected/input'),
                    "dpred": os.path.join(base_pred, 'multi_biasfield/target_corrected/pred'),
                    "dtarget": os.path.join(base_pred, 'multi_biasfield/target_corrected/target'),
                    "dmask": sel_mask_dir,
                    "dmask_fat": fat_mask_dir,
                    "name": "test"}

# Using single as input.. to avoid any differences...
multi_homog_test = {"dinput": os.path.join(base_pred, 'single_homogeneous/target_corrected/input'),
                    "dpred": os.path.join(base_pred, 'multi_homogeneous/target_corrected/pred'),
                    "dtarget": os.path.join(base_pred, 'multi_homogeneous/target_corrected/target'),
                    "dmask": sel_mask_dir,
                    "dmask_fat": fat_mask_dir,
                    "name": "test"}

n4itk_test = {"dinput": input_n4_test,
              "dpred": "/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/corrected_N4",
              "dtarget": "/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/target",
              "dmask": sel_mask_dir,
              "dmask_fat": fat_mask_dir,
              "name": "test"}

# Volunteer set
volunteer_list = [single_biasf_volunteer, single_homog_volunteer, multi_biasf_volunteer, multi_homog_volunteer]
# Patient 3T set
patient_3T_list = [single_biasf_patient_3T, single_homog_patient_3T]
# Patient 7T set
patient_7T_list = [single_biasf_patient, single_homog_patient]
# Test set
test_list = [single_biasf_test, multi_biasf_test, single_homog_test, multi_homog_test, n4itk_test]
# test_list = [single_homog_test, multi_homog_test]

# Lets test test thing..
# Grab a file..
# Visualize it
# Get target
# Demonstrate error
input_dir = test_list[0]['dinput']
target_dir = test_list[0]['dtarget']
mask_dir = test_list[0]['dmask']
file_list = os.listdir(target_dir)
sel_file = file_list[0]

target_file = os.path.join(target_dir, sel_file)
mask_file = os.path.join(mask_dir, sel_file)
input_file = os.path.join(input_dir, sel_file)

target_array = hmisc.load_array(target_file).T[:, ::-1, ::-1]
mask_array = hmisc.load_array(mask_file).T[:, ::-1, ::-1]
input_array = hmisc.load_array(input_file).T[:, ::-1, ::-1]

n_slice = target_array.shape[0]
sel_target_array = target_array[n_slice // 2]
sel_mask_array = mask_array[n_slice // 2]
sel_input_array = input_array[n_slice // 2]

# Mask the target array...
sel_target_array = np.ma.masked_array(sel_target_array, mask=1-sel_mask_array)
sel_input_array = np.ma.masked_array(sel_input_array, mask=1-sel_mask_array)
hist_target, _ = np.histogram(sel_target_array.ravel(), bins=256, range=(0, 255), density=True)
img_array = {'target': sel_target_array}
rmse_values = {'target': '0'}
wasserstein_values = {'target': '0'}
ssim_values = {'target': '1'}
for i_dict in test_list:
    model_name = os.path.basename(os.path.dirname(os.path.dirname(i_dict['dpred'])))
    print(model_name)
    pred_file = os.path.join(i_dict['dpred'], sel_file)
    loaded_array = hmisc.load_array(pred_file).T[:, ::-1, ::-1]
    n_slice = loaded_array.shape[0]
    sel_array = loaded_array[n_slice//2]
    sel_array = np.ma.masked_array(sel_array, mask=1 - sel_mask_array)
    c_scale = (np.mean(sel_target_array) / np.mean(sel_array))
    sel_array = sel_array * c_scale
    img_array[model_name] = sel_array
    rmse_values[model_name] = np.sqrt(np.mean((sel_array - sel_target_array) ** 2))
    hist_pred, _ = np.histogram(sel_array.ravel(), bins=256, range=(0, 255), density=True)
    wasserstein_values[model_name] = scipy.stats.wasserstein_distance(hist_pred, hist_target)
    ssim_values[model_name] = skimage.metrics.structural_similarity(sel_array, sel_target_array)

hmisc.print_dict(ssim_values)
hmisc.print_dict(wasserstein_values)
hmisc.print_dict(rmse_values)
fig_obj = hplotc.ListPlot(np.array(list(img_array.values())) * sel_mask_array)
fig_obj.figure.savefig('/local_scratch/sharreve/test.png')

import small_project.homogeneity_measure.metric_implementations as hhomog
# Now check the GLCM with and without masking...
img_array = {'target': sel_input_array}
# Get target vs input
# Now we will scale with input...
c_scale = (np.mean(sel_input_array) / np.mean(sel_target_array))
glcm_features = hhomog.get_relative_glcm_features(sel_target_array * c_scale, sel_input_array, glcm_dist=list(range(7))[1:])
glcm_array = {'target': glcm_features}
for i_dict in test_list:
    model_name = os.path.basename(os.path.dirname(os.path.dirname(i_dict['dpred'])))
    print(model_name)
    pred_file = os.path.join(i_dict['dpred'], sel_file)
    loaded_array = hmisc.load_array(pred_file).T[:, ::-1, ::-1]
    n_slice = loaded_array.shape[0]
    sel_array = loaded_array[n_slice//2]
    sel_array = np.ma.masked_array(sel_array, mask=1 - sel_mask_array)
    # Now we will scale with input...
    c_scale = (np.mean(sel_input_array) / np.mean(sel_array))
    sel_array = sel_array * c_scale
    img_array[model_name] = sel_array
    glcm_features = hhomog.get_relative_glcm_features(sel_array, sel_input_array, glcm_dist=list(range(7))[1:])
    glcm_array[model_name] = glcm_features

hmisc.print_dict(glcm_array)