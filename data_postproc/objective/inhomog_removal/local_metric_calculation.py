"""
Okay.. so we got some GLCM metric...

We want to see `before` and `after`

I want to check CoV over tissue types as well

All metrics should be caclulated with masks

I can report it in a similar manner as the dice score..
"""

import json
import helper.misc as hmisc
import helper.array_transf as harray
import small_project.homogeneity_measure.metric_implementations as homog_metric
import os
import numpy as np
import matplotlib.pyplot as plt
import helper.plot_class as hplotc
import skimage.feature
import re
import objective.inhomog_removal.CalculateMetrics as CalcMetrics


if __name__ == "__main__":
    """
    Define all paths for volunteer data..
    """
    body_mask_dir = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/body_mask'
    prostate_mask_dir = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/prostate_mask'
    fat_mask_dir = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/subcutaneous_fat_mask'
    muscle_mask_dir = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/muscle_mask'

    for sel_mask_dir in [body_mask_dir, prostate_mask_dir, fat_mask_dir, muscle_mask_dir][0:1]:

        multi_biasf_volunteer = {"dinput": "/media/bugger/MyBook/data/paper/inhomog_removal/volunteer/multi_biasfield/input",
                                "dtarget": "/media/bugger/MyBook/data/paper/inhomog_removal/volunteer/multi_biasfield/pred",
                                "dmask": sel_mask_dir}

        single_biasf_volunteer = {"dinput": "/media/bugger/MyBook/data/paper/inhomog_removal/volunteer/single_biasfield/input",
                                "dtarget": "/media/bugger/MyBook/data/paper/inhomog_removal/volunteer/single_biasfield/pred",
                                "dmask": sel_mask_dir}

        multi_homog_volunteer = {"dinput": "/media/bugger/MyBook/data/paper/inhomog_removal/volunteer/multi_homogeneous/input",
                                "dtarget": "/media/bugger/MyBook/data/paper/inhomog_removal/volunteer/multi_homogeneous/pred",
                                "dmask": sel_mask_dir}

        single_homog_volunteer = {"dinput": "/media/bugger/MyBook/data/paper/inhomog_removal/volunteer/single_homogeneous/input",
                              "dtarget": "/media/bugger/MyBook/data/paper/inhomog_removal/volunteer/single_homogeneous/pred",
                              "dmask": sel_mask_dir}

        n4itk_volunteer = {"dinput": "/media/bugger/MyBook/data/7T_data/prostate_t2_selection/t2w_n4itk/input",
                            "dtarget": "/media/bugger/MyBook/data/7T_data/prostate_t2_selection/t2w_n4itk/pred",
                            "dmask": sel_mask_dir}

        loop_this_stuff = [multi_biasf_volunteer, single_biasf_volunteer, multi_homog_volunteer, single_homog_volunteer, n4itk_volunteer]

        for i_dict in loop_this_stuff:
            print(f"Input path {i_dict['dinput']}")
            mask_dir_name = os.path.basename(i_dict['dmask'])
            mask_name = re.sub('_mask', '', mask_dir_name)
            dest_dir = os.path.dirname(i_dict['dinput'])
            metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.npy', patch_size=64, **i_dict)
            glcm_relative_change = metric_obj.run_glcm_features()
            glcm_relative_change_dict = hmisc.listdict2dictlist(glcm_relative_change)
            cov_values = metric_obj.run_joint_cov()
            np.save(os.path.join(dest_dir, f'{mask_name}_coef_of_variation.npy'), cov_values)
            relative_change_ser = json.dumps(glcm_relative_change_dict)
            with open(os.path.join(dest_dir, f'{mask_name}_rel_change_glcm.json'), 'w') as f:
                f.write(relative_change_ser)


    """
    Calc patient data
    """
    n4itk_patient = {
        "dinput": "/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/image_n4itk/input",
        "dtarget": "/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/image_n4itk/pred",
        "dmask": "/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/mask"}

    single_homog_patient = {
        "dinput": "/media/bugger/MyBook/data/paper/inhomog_removal/patient/single_homogeneous/input",
        "dtarget": "/media/bugger/MyBook/data/paper/inhomog_removal/patient/single_homogeneous/pred",
        "dmask": "/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/mask"}

    single_biasf_patient = {
        "dinput": "/media/bugger/MyBook/data/paper/inhomog_removal/patient/single_biasfield/input",
        "dtarget": "/media/bugger/MyBook/data/paper/inhomog_removal/patient/single_biasfield/pred",
        "dmask": "/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/mask"}


    loop_this_too = [n4itk_patient, single_homog_patient, single_biasf_patient]

    for i_dict in loop_this_too:
        dest_dir = os.path.dirname(i_dict['dinput'])
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.npy', patch_size=256, **i_dict)
        glcm_relative_change = metric_obj.run_glcm_features()
        glcm_relative_change_dict = hmisc.listdict2dictlist(glcm_relative_change)
        cov_values = metric_obj.run_joint_cov()
        np.save(os.path.join(dest_dir, f'{mask_name}_coef_of_variation.npy'), cov_values)
        relative_change_ser = json.dumps(glcm_relative_change_dict)
        with open(os.path.join(dest_dir, f'{mask_name}_rel_change_glcm.json'), 'w') as f:
            f.write(relative_change_ser)