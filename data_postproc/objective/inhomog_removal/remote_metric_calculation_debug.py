import os
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=6
from multiprocessing import set_start_method
set_start_method("spawn")
import json
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
import small_project.homogeneity_measure.metric_implementations as homog_metric
base_pred = '/local_scratch/sharreve/model_run/selected_inhomog_removal_models'

"""
We want to debug some stuff...
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
volunteer_list = [single_biasf_volunteer, single_homog_volunteer, multi_biasf_volunteer, multi_homog_volunteer, n4itk_volunteer]
# Patient 3T set
patient_3T_list = [single_biasf_patient_3T, single_homog_patient_3T, n4itk_patient_3T]
# Patient 7T set
patient_7T_list = [single_biasf_patient, single_homog_patient, n4itk_patient]
# Test set
test_list = [single_biasf_test, multi_biasf_test, single_homog_test, multi_homog_test, n4itk_test]
# test_list = [single_homog_test, multi_homog_test]

# These two differ a LOT. Lets double check everything
for i_dict in [single_biasf_test]:  # test_list + volunteer_list + patient_7T_list + patient_3T_list:
    dataset_name = i_dict['name']
    print(f"Input path {i_dict['dinput']}")
    mask_dir_name = os.path.basename(i_dict['dmask'])
    model_name = os.path.basename(os.path.dirname(os.path.dirname(i_dict['dpred'])))
    mask_name = re.sub('_mask', '', mask_dir_name)
    dest_dir = os.path.dirname(i_dict['dpred'])
    if dataset_name == 'volunteer':
        # Needed for volunteer 7T data
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.npy', patch_size=10*10, **i_dict)
        metric_obj.glcm_dist = list(range(10))[1:]
    elif dataset_name == 'patient_3T':
        # Needed for patient 3T
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.h5', patch_size=7*10, mask_suffix='_target', **i_dict)
        metric_obj.glcm_dist = list(range(7))[1:]
    elif dataset_name == 'patient':
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.npy', patch_size=16*10, **i_dict)
        # We want 5mm and we have a pixel spacing of approx 0.28mm
        metric_obj.glcm_dist = list(range(16))[1:]
    elif dataset_name == 'test':
        # In the test set we have a pixel spacing of...
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.nii.gz', patch_size=7*10, **i_dict)
        # We want 5mm and we have a pixel spacing of approx 0.7mm
        metric_obj.glcm_dist = list(range(7))[1:]
    print(" ONLY USING THREE FILES TO MAKE SURE THAT WE CAN QUICKLY CHECK SOME RESULTS")
    metric_obj.file_list = metric_obj.file_list[0:3]
    metric_obj.load_file(2)
    metric_obj.set_slice(metric_obj.n_slices // 2)
    input_slice = metric_obj.loaded_image_slice
    pred_slice = metric_obj.loaded_pred_slice
    mask_slice = metric_obj.loaded_mask_slice
    file_name = f'{dataset_name}_{model_name}'
    metric_obj.save_current_slice("normal/" +file_name)
    print(f"=========================={file_name}==============================")
    print("\n\nNormal")
    # metric_obj.print_features_current_slice()
    # metric_obj.print_target_features_current_slice()
    import cv2
    nx, ny = metric_obj.loaded_mask_slice.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(nx // 32, ny // 32))
    metric_obj.loaded_image_slice = clahe.apply(metric_obj.loaded_image_slice)
    metric_obj.loaded_pred_slice = clahe.apply(metric_obj.loaded_pred_slice)
    if metric_obj.dtarget is not None:
        metric_obj.loaded_target_slice = clahe.apply(metric_obj.loaded_target_slice)
    metric_obj.save_current_slice("with_clahe/" + file_name)
    print("\n\nSingle CLAHE")
    # metric_obj.print_features_current_slice()
    # metric_obj.print_target_features_current_slice()
    # Another one
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(nx // 128, ny // 128))
    metric_obj.loaded_pred_slice = harray.treshold_percentile(metric_obj.loaded_pred_slice, q=98)
    metric_obj.loaded_pred_slice = clahe.apply(metric_obj.loaded_pred_slice)
    metric_obj.save_current_slice("second_clahe/" + file_name)
    print("\n\nSecond clahe")
    # metric_obj.print_features_current_slice()
    # metric_obj.print_target_features_current_slice()
    hplotc.close_all()
    fuzzy_input_dict = homog_metric.get_fuzzy_features(metric_obj.loaded_image_slice, patch_size=metric_obj.slice_patch_size)
    fuzzy_pred_dict = homog_metric.get_fuzzy_features(metric_obj.loaded_pred_slice, patch_size=metric_obj.slice_patch_size)
    fuzzy_target_dict = homog_metric.get_fuzzy_features(metric_obj.loaded_target_slice,patch_size=metric_obj.slice_patch_size)
    print('------input')
    hmisc.print_dict(fuzzy_input_dict)
    print('------pred')
    hmisc.print_dict(fuzzy_pred_dict)
    print('------target')
    hmisc.print_dict(fuzzy_target_dict)
