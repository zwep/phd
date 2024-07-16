import numpy as np
import os
import json
import helper.plot_fun as hplotf
import pandas as pd
import csv
import helper.misc as hmisc
import papers.inhomog_removal.helper_inhomog as helper_inhomog
from objective_configuration.inhomog_removal import IMG_SYNTH_COIL, get_path_dict, MODEL_DIR, CHOSEN_FEATURE
import argparse
import helper.array_transf as harray
import re
import sys
from loguru import logger

"""
Okay so we calculated some coefficient of variation..
and we want to display them

We also got the GLCM measures..
"""

parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, help='Provide the name of the directory that we want to post process')
parser.add_argument('-dataset', type=str, default='all',
                    help='Provide the name of the dataset on which we want to evaluate: '
                         'synthetic, 3T, patient, volunteer')
parser.add_argument('--debug', default=False, action='store_true')

parser.add_argument('--glcm', default=False, action='store_true')
parser.add_argument('--cov', default=False, action='store_true')
parser.add_argument('--target', default=False, action='store_true')
parser.add_argument('--input', default=False, action='store_true')


p_args = parser.parse_args()
path = p_args.path
dataset = p_args.dataset
debug = p_args.debug

glcm_metric = p_args.glcm
cov_metric = p_args.cov
target_metric = p_args.target
input_metric = p_args.input

# Get all paths..
dconfig = os.path.join(MODEL_DIR, path)
path_dict, dataset_list = get_path_dict(dconfig)

# Check what kind of synthetic data input we need (multi coil or single)
single_input = True
if 'multi' in path:
    single_input = False
    # Use the individual coil images path as input
    # Why..?
    # path_dict['synthetic']['dimage'] = IMG_SYNTH_COIL
    # Limit the number of datasets we check
    dataset_list = ['volunteer', 'synthetic']

if dataset == 'all':
    sel_dataset_list = dataset_list
else:
    if dataset in dataset_list:
        sel_dataset_list = [dataset]
    else:
        logger.debug(f'Unknown dataset selected: {dataset}')
        sys.exit()


group_names = ['patient', 'patient_3T', 'volunteer', 'synthetic']
glcm_file_name_list = ['input_change_glcm', 'pred_change_glcm', 'target_change_glcm']
cov_file_name_list = ['rel_coef_of_variation', 'rel_target_coef_of_variation']


"""
Print the relative GLCM values between the 
"""

if glcm_metric:
    # Here we collect all the files and put it in a single dictionary
    glcm_metric_dict = {}
    for glcm_file_name in glcm_file_name_list:
        logger.info(f'Collecting GLCM file {glcm_file_name} ')
        dict_key = glcm_file_name.split("_")[0]
        temp_metric_dict = {}
        for i_dataset in sel_dataset_list:
            logger.info(f'\t With dataset {i_dataset} ')
            path_to_pred = path_dict[i_dataset]['dpred']
            temp_glcm_dict = helper_inhomog.collect_glcm(path_to_pred, file_name=glcm_file_name,
                                                         temp_dict=temp_metric_dict, key=i_dataset)
        glcm_metric_dict[dict_key] = temp_metric_dict

    if debug:
        logger.debug(f'Here is the collected GLCM dict')
        hmisc.print_dict(glcm_metric_dict)

    # Here we collect for each data type, model, and feature the mean+std
    glcm_aggregate_dict = {}
    for data_type, model_dict in glcm_metric_dict.items():
        logger.info(f'Aggregating GLCM data type {data_type}')
        _ = glcm_aggregate_dict.setdefault(data_type, {})
        for dataset_name, metric_dict in model_dict.items():
            logger.info(f'\t And dataset name {dataset_name}')
            _ = glcm_aggregate_dict[data_type].setdefault(dataset_name, {})
            if 'file_list' in metric_dict:
                del metric_dict['file_list']
                del metric_dict['slice_list']
            for metric_key in CHOSEN_FEATURE:
                logger.info(f'\t\t And aggregating metric {metric_key}')
                metric_value = metric_dict[metric_key]
                _ = glcm_aggregate_dict[data_type][dataset_name].setdefault(metric_key, {})
                glcm_aggregate_dict[data_type][dataset_name][metric_key] = helper_inhomog.get_mean_std_str(metric_value)
                # If we are calculating stuff for prediction, we can immediatly  calculate the relative stuff with the input
                if data_type == 'pred':
                    metric_value_input = np.array(glcm_metric_dict['input'][dataset_name][metric_key])
                    metric_value_pred = np.array(metric_value)
                    _ = glcm_aggregate_dict.setdefault('rel_pred', {})
                    _ = glcm_aggregate_dict['rel_pred'].setdefault(dataset_name, {})
                    _ = glcm_aggregate_dict['rel_pred'][dataset_name].setdefault(metric_key, {})
                    glcm_aggregate_dict['rel_pred'][dataset_name][metric_key] = helper_inhomog.get_mean_std_str(
                        (metric_value_pred - metric_value_input) / metric_value_input)
                # Same thing for the target...
                if data_type == 'target':
                    metric_value_input = np.array(glcm_metric_dict['input'][dataset_name][metric_key])
                    metric_value_target = np.array(metric_value)
                    _ = glcm_aggregate_dict.setdefault('rel_target', {})
                    _ = glcm_aggregate_dict['rel_target'].setdefault(dataset_name, {})
                    _ = glcm_aggregate_dict['rel_target'][dataset_name].setdefault(metric_key, {})
                    glcm_aggregate_dict['rel_target'][dataset_name][metric_key] = helper_inhomog.get_mean_std_str(
                        (metric_value_target - metric_value_input) / metric_value_input)

    # Since we also want to calculate relative features, which are values OVER two different data types
    data_type_list = list(glcm_aggregate_dict.keys())
    for i_data_type in data_type_list:
        logger.info(i_data_type)
        temp_df = pd.DataFrame(glcm_aggregate_dict[i_data_type])
        helper_inhomog.print_dataframe(temp_df.T)

"""
Print the Coefficient of variation values.. (rel, inp, pred)
"""

if cov_metric:
    coefv_metric_dict = {}
    for cov_file_name in cov_file_name_list:
        logger.info(f'Aggregating coef of var data type {cov_file_name}')
        res_dict_coefv = {}
        for i_dataset in sel_dataset_list:
            logger.info(f'\t And dataset {i_dataset}')
            path_to_pred = path_dict[i_dataset]['dpred']
            res_dict_coefv = helper_inhomog.collect_a_npy_file(path_to_pred, file_name=cov_file_name,
                                                               temp_dict=res_dict_coefv, key=i_dataset)
        coefv_metric_dict[cov_file_name] = res_dict_coefv


    coefv_aggregate_dict = {}
    for data_type, res_dict_coefv in coefv_metric_dict.items():
        logger.info(f'Aggregating data type {data_type}')
        _ = coefv_aggregate_dict.setdefault(data_type, {})
        # relative COV stuff
        aggregate_coefv = harray.aggregate_dict_mean_value(res_dict_coefv, agg_dict={})
        coefv_aggregate_dict[data_type]['coef of variation'] = aggregate_coefv

    data_type_list = list(coefv_aggregate_dict.keys())
    for i_data_type in data_type_list:
        logger.info(f'Data type {i_data_type}')
        temp_df = pd.DataFrame(coefv_aggregate_dict[i_data_type])
        helper_inhomog.print_dataframe(temp_df)

"""
Print the RMSE values...
"""

if target_metric:
    res_dict_rmse = {}
    for i_dataset in sel_dataset_list:
        path_to_pred = path_dict[i_dataset]['dpred']
        res_dict_rmse = helper_inhomog.collect_a_npy_file(path_to_pred, 'rmse_values',
                                                          temp_dict=res_dict_rmse, key=i_dataset)

    rmse_aggregate_dict = {}
    for i_dataset, rmse_value_list in res_dict_rmse.items():
        rmse_aggregate_dict.setdefault(i_dataset, {})
        rmse_aggregate_dict[i_dataset]['rmse'] = helper_inhomog.get_mean_std_str(rmse_value_list)

    rmse_df = pd.DataFrame(rmse_aggregate_dict)
    helper_inhomog.print_dataframe(rmse_df.T)


    """
    Print the SSIM values...
    """

    res_dict_ssim = {}
    for i_dataset in sel_dataset_list:
        path_to_pred = path_dict[i_dataset]['dpred']
        res_dict_ssim = helper_inhomog.collect_a_npy_file(path_to_pred, 'ssim_values', temp_dict=res_dict_ssim, key=i_dataset)


    ssim_aggregate_dict = {}
    for i_dataset, v in res_dict_ssim.items():
        ssim_aggregate_dict.setdefault(i_dataset, {})
        v = np.array(v)
        v[v>1] = 1
        v[v<0] = 0
        ssim_aggregate_dict[i_dataset]['ssim'] = helper_inhomog.get_mean_std_str(v)

    ssim_df = pd.DataFrame(ssim_aggregate_dict)
    helper_inhomog.print_dataframe(ssim_df.T)

    """
    Overview Wasserstein
    """

    res_dict_wass = {}
    for i_dataset in sel_dataset_list:
        path_to_pred = path_dict[i_dataset]['dpred']
        res_dict_wass = helper_inhomog.collect_a_npy_file(path_to_pred,
                                                          'wasserstein_values',
                                                          temp_dict=res_dict_wass, key=i_dataset)

    wd_aggregate_dict = {}
    for i_dataset, v in res_dict_wass.items():
        wd_aggregate_dict.setdefault(i_dataset, {})
        if len(v):
            wd_aggregate_dict[i_dataset]['wasserstein'] = helper_inhomog.get_mean_std_str(np.array(v) * 100)

    wd_df = pd.DataFrame(wd_aggregate_dict)
    helper_inhomog.print_dataframe(wd_df.T)


"""
Print the RMSE values... between input and target
"""

if input_metric:
    res_dict_rmse = {}
    for i_dataset in sel_dataset_list:
        path_to_pred = path_dict[i_dataset]['dpred']
        res_dict_rmse = helper_inhomog.collect_a_npy_file(path_to_pred, 'input_rmse',
                                                          temp_dict=res_dict_rmse, key=i_dataset)

    rmse_aggregate_dict = {}
    for i_dataset, rmse_value_list in res_dict_rmse.items():
        rmse_aggregate_dict.setdefault(i_dataset, {})
        rmse_aggregate_dict[i_dataset]['rmse'] = helper_inhomog.get_mean_std_str(rmse_value_list)

    rmse_df = pd.DataFrame(rmse_aggregate_dict)
    helper_inhomog.print_dataframe(rmse_df.T)


    """
    Print the SSIM values.... between input and target
    """

    res_dict_ssim = {}
    for i_dataset in sel_dataset_list:
        path_to_pred = path_dict[i_dataset]['dpred']
        res_dict_ssim = helper_inhomog.collect_a_npy_file(path_to_pred, 'input_ssim', temp_dict=res_dict_ssim, key=i_dataset)


    ssim_aggregate_dict = {}
    for i_dataset, v in res_dict_ssim.items():
        ssim_aggregate_dict.setdefault(i_dataset, {})
        v = np.array(v)
        v[v>1] = 1
        v[v<0] = 0
        ssim_aggregate_dict[i_dataset]['ssim'] = helper_inhomog.get_mean_std_str(v)

    ssim_df = pd.DataFrame(ssim_aggregate_dict)
    helper_inhomog.print_dataframe(ssim_df.T)

    """
    Overview Wasserstein. between input and target
    """

    res_dict_wass = {}
    for i_dataset in sel_dataset_list:
        path_to_pred = path_dict[i_dataset]['dpred']
        res_dict_wass = helper_inhomog.collect_a_npy_file(path_to_pred,
                                                          'input_wasserstein',
                                                          temp_dict=res_dict_wass, key=i_dataset)

    wd_aggregate_dict = {}
    for i_dataset, v in res_dict_wass.items():
        wd_aggregate_dict.setdefault(i_dataset, {})
        if len(v):
            wd_aggregate_dict[i_dataset]['wasserstein'] = helper_inhomog.get_mean_std_str(np.array(v) * 100)

    wd_df = pd.DataFrame(wd_aggregate_dict)
    helper_inhomog.print_dataframe(wd_df.T)

#
#
# for k, v in res_dict_ssim.items():
#     print(k, v.min(), v.max())
#
# for k, v in res_dict_wass.items():
#     print(k, v.min(), v.max())
#
# temp_glcm_dict = {}
# for k, v in RESULT_PATH.items():
#     temp_glcm_dict = helper_inhomog.collect_glcm(v, file_name='rel_change_glcm', temp_dict=temp_glcm_dict, key=k)
#
# inp_glcm_dict = {}
# for k, v in RESULT_PATH.items():
#     inp_glcm_dict = helper_inhomog.collect_glcm(v, file_name='input_change_glcm', temp_dict=inp_glcm_dict, key=k)
#
# pred_glcm_dict = {}
# for k, v in RESULT_PATH.items():
#     pred_glcm_dict = helper_inhomog.collect_glcm(v, file_name='pred_change_glcm', temp_dict=pred_glcm_dict, key=k)
#
# tgt_glcm_dict = {}
# for k, v in RESULT_PATH.items():
#     tgt_glcm_dict = helper_inhomog.collect_glcm(v, file_name='target_change_glcm', temp_dict=tgt_glcm_dict, key=k)
#
#
# import matplotlib.pyplot as plt
# sel_model = 'single_biasf_test'
# for sel_model in res_dict_ssim.keys():
#     print(sel_model)
#     # glcm_metric = temp_glcm_dict[sel_model]['energy']#[5::10]
#     glcm_metric = (np.array(pred_glcm_dict[sel_model]['energy']) - np.array(inp_glcm_dict[sel_model]['energy'])) / inp_glcm_dict[sel_model]['energy']
#     ssim_value = res_dict_ssim[sel_model]#[5::10]
#     mask_indices = (ssim_value < 0).astype(int) + (ssim_value > 1).astype(int)
#     mask_indices = mask_indices.astype(bool)
#     #
#     wd_value = res_dict_wass[sel_model]#[5::10]
#     rmse_value = res_dict_rmse[sel_model]#[5::10]
#     #
#     ssim_value = np.ma.masked_where(mask_indices, ssim_value)
#     rmse_value = np.ma.masked_where(mask_indices, rmse_value)
#     wd_value = np.ma.masked_where(mask_indices, wd_value)
#     glcm_metric = np.ma.masked_where(mask_indices, glcm_metric)
#     fig, ax = plt.subplots(3, figsize=(5, 15))
#     fig.suptitle('GLCM vs ...')
#     ax[0].scatter(glcm_metric, ssim_value, label='ssim')
#     ax[0].set_title(f"Correlation {np.round(np.corrcoef(glcm_metric, ssim_value)[0, 1], 2)}")
#     ax[1].scatter(glcm_metric, wd_value, label='wd')
#     ax[1].set_title(f"Correlation {np.round(np.corrcoef(glcm_metric, wd_value)[0, 1], 2)}")
#     ax[2].scatter(glcm_metric, rmse_value, label='rmse')
#     ax[2].set_title(f"Correlation {np.round(np.corrcoef(glcm_metric, rmse_value)[0, 1], 2)}")
#     [temp_ax.legend() for temp_ax in ax]
#     fig.savefig(f'/local_scratch/sharreve/{sel_model}_cor_glcm.png')
#     #
#     fig, ax = plt.subplots(2, figsize=(5, 15))
#     fig.suptitle('WD vs ...')
#     ax[0].scatter(wd_value, ssim_value, label='ssim')
#     ax[0].set_title(f"Correlation {np.round(np.corrcoef(wd_value, ssim_value)[0, 1], 2)}")
#     ax[1].scatter(wd_value, rmse_value, label='rmse')
#     ax[1].set_title(f"Correlation {np.round(np.corrcoef(wd_value, rmse_value)[0, 1], 2)}")
#     [temp_ax.legend() for temp_ax in ax]
#     fig.savefig(f'/local_scratch/sharreve/{sel_model}_cor_wd.png')
#     #
#     fig, ax = plt.subplots(1, figsize=(5, 15))
#     fig.suptitle('SSIM vs RMSE')
#     ax.scatter(ssim_value, rmse_value, label='rmse')
#     ax.set_title(f"Correlation {np.round(np.corrcoef(ssim_value, rmse_value)[0, 1], 2)}")
#     ax.legend()
#     fig.savefig(f'/local_scratch/sharreve/{sel_model}_cor_ssim.png')
#
# """
# Lets check how the iterative recon worked...
# """
#
# # abs_iter_dict = {}
# # for k, v in RESULT_PATH.items():
# #     print(v)
# #     abs_iter_dict = collect_a_csv_file(v, file_name='abs_iterative', temp_dict=abs_iter_dict, key=k)
# #
# # for k, v in abs_iter_dict.items():
# #     fig_obj = hplotf.plot_multi_lines(v)
# #     fig_obj.suptitle(k)
# #     fig_obj.savefig(f'/local_scratch/sharreve/{k}.png')
#
# rel_iter_dict = {}
# for k, v in RESULT_PATH.items():
#     rel_iter_dict = helper_inhomog.collect_a_csv_file(v, file_name='rel_iterative', temp_dict=rel_iter_dict, key=k)
#
# hplotf.close_all()
#
# # color_dict = {'patient_3T': "#FF9F29",
# #               'patient': "#753422",
# #               'volunteer': "#73A9AD",
# #               'test': "#243A73"}
# color_dict = {'patient_3T': "#E01717",
#               'patient': "#192AE0",
#               'volunteer': "#74E393",
#               'test': "#E0B519"}
#
# import matplotlib.pyplot as plt
# fig_dict = {'single_homog': (plt.subplots()),
#             'multi_homog': (plt.subplots()),
#             'single_biasf': (plt.subplots()),
#             'multi_biasf': (plt.subplots())}
#
# name_converter = {'single_homog': 'Single channel t-Image',
#                   'multi_homog': 'Multi channel t-Image',
#                   'single_biasf': 'Single channel t-Biasfield',
#                   'multi_biasf': 'Multi channel t-Biasfield'}
#
# dataset_name_converter = {'test': 'Test split',
#                           'patient_3T': 'Patient data 3T',
#                           'patient': 'Patient data 7T',
#                           'volunteer': 'Volunteer data 7T'}
# import re
# params = {'mathtext.default': 'regular'}
# font_size = 18
# plt.rcParams.update(params)
# for k, v in rel_iter_dict.items():
#     sel_color = '#0F0E0E'
#     k_without_dataset = re.sub('|'.join(["_" + x for x in color_dict.keys()]), '', k)
#     sel_fig, sel_ax = fig_dict[k_without_dataset]
#     for sel_dataset, sel_color in color_dict.items():
#         if k.endswith(sel_dataset):
#             break
#     std_1 = np.std(v, axis=1)
#     mean_value = np.mean(v, axis=1)
#     # This should be changed..?
#     sel_ax.plot(mean_value, sel_color, label=dataset_name_converter[sel_dataset])
#     sel_ax.fill_between(np.arange(len(v)), mean_value-std_1, mean_value+std_1, color=sel_color, alpha=0.3)
#     sel_ax.set_xlabel('Number of iterations', fontsize=font_size)
#     sel_ax.set_ylabel('$\|\| y_i - y_{i+1}\|\|$', fontsize=font_size)
#     sel_fig.suptitle(name_converter[k_without_dataset], fontsize=font_size)
#
# for k_plot, v_plot in fig_dict.items():
#     sel_fig, sel_ax = v_plot
#     sel_ax.legend()
#     sel_fig.savefig(f'/local_scratch/sharreve/{k_plot}.png', bbox_inches='tight')
#
#
# """
# Below we can manually calculate stuff on the test split
# """

#
# input_lines = []
# model_lines = []
# result_lines = []
# target_lines = []
# # for i_model in ['multi_biasfield', 'multi_homogeneous', 'single_biasfield', 'single_homogeneous']:
# print(i_model)
# ddatapred = f'/local_scratch/sharreve/model_run/selected_inhomog_removal_models/{i_model}/target_corrected/pred'
# ddatatarget = f'/local_scratch/sharreve/model_run/selected_inhomog_removal_models/{i_model}/target_corrected/target'
# ddatainp = f'/local_scratch/sharreve/model_run/selected_inhomog_removal_models/{i_model}/target_corrected/input'
# ddatamask = f'/local_scratch/sharreve/model_run/selected_inhomog_removal_models/{i_model}/target_corrected/mask'
# # ddatamask = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/mask'
# # ddatapred = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/corrected_N4'
# # ddatatarget = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/target'
# s = 0
# ssim_value = 0
# counter = 0
# i_file = os.listdir(ddatapred)[0]
# # for i_file in os.listdir(ddatapred)[0:1]:
# H = hmisc.load_array(os.path.join(ddatainp, i_file)).T[:, ::-1, ::-1]
# I = hmisc.load_array(os.path.join(ddatapred, i_file)).T[:, ::-1, ::-1]
# J = hmisc.load_array(os.path.join(ddatatarget, i_file)).T[:, ::-1, ::-1]
# Mask = hmisc.load_array(os.path.join(ddatamask, i_file)).T[:, ::-1, ::-1]
# # I = harray.scale_minmax(I)
# # H = harray.scale_minmax(H)
# # for i_slice in range(n_slice):
# n_slice, nx, ny = I.shape
# # i_slice = n_slice//2
# i_slice = 10
# counter += 1
# M_img = harray.scale_minmax(Mask[i_slice]).astype(np.uint8)
# M_sub_img = harray.create_random_center_mask(M_img.shape, random=False, mask_fraction=0.07)
# H_img = H[i_slice]
# H_line = H_img[nx//2]
# H_sub_img = H_img * M_sub_img
# I_img = I[i_slice]
# I_line = I_img[nx//2]
# I_sub_img = I_img * M_sub_img
# J_img = J[i_slice]
# J_line = J_img[nx // 2]
# J_sub_img = J_img * M_sub_img
# # Perform it with I and J [target]
# import small_project.homogeneity_measure.metric_implementations as homog_metric
# # Or with I and H [input]
# minimize_obj = homog_metric.MinimizeL2Line(I_sub_img, J_sub_img)
# result_min = minimize_obj.minimize_run()
# a_slope, b_offset = result_min
# # Scale manually...
# scale_I_to_J = np.mean(J_sub_img[J_sub_img!=0]) / np.mean(I_sub_img[I_sub_img!=0])
# I_img_scaled = I_img * scale_I_to_J
# for a, b in [[scale_I_to_J, 0], [a_slope, b_offset]]:
#     #I_img_scaled = minimize_obj.get_optimal_transform()
#     # Or with I and H [input] - image version
#     # minimize_obj = MinimizeL2(I_sub_img, H_sub_img)
#     # Or with I and J [target] - image version
#     #    minimize_obj = homog_metric.MinimizeL2(I_line, J_line)
#     # model_lines.append(I_line)
#     # a, b = [ 0.671203792921229, -0.07014338922578037]
#     I_img_scaled = I_img * a + b
#     result_line = I_img_scaled[nx//2]
#     input_lines.append(H_line)
#     result_lines.append(result_line)
#     target_lines.append(J_line)
#     wasserstein_distance_target_pred = scipy.stats.wasserstein_distance((I_img_scaled[M_img==1]).ravel(), (J_img[M_img==1]).ravel())
#     ssim_target_pred = structural_similarity(harray.scale_minmax(M_img*I_img_scaled), harray.scale_minmax(M_img*J_img))
#     rms = mean_squared_error(M_img*I_img_scaled, M_img*J_img, squared=False)
#     print(wasserstein_distance_target_pred)
#     print(ssim_target_pred)
#     print(rms)
#     print()
