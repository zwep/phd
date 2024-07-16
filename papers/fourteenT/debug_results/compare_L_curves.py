from objective_helper.fourteenT import VisualizeAllMetrics, DataCollector
import matplotlib.pyplot as plt
import scipy.io
import helper.misc as hmisc
import os
import objective_helper.fourteenT as helper_14T
from objective_configuration.fourteenT import COIL_NAME_ORDER, COLOR_DICT, \
    DPLOT_1KT_BETA_POWER, DDATA_1KT_BETA_POWER, \
    DPLOT_1KT_BETA_VOP, DDATA_1KT_BETA_VOP, \
    DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP, \
    DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER, \
    COIL_NAME_ORDER_TRANSLATOR, RF_SCALING_FACTOR_1KT, WEIRD_RF_FACTOR, TARGET_FLIP_ANGLE, SUBDIR_RANDOM_SHIM
import os
import numpy as np

"""
Here we are going to compare L-curves from the different simulations

Just to check if everything has gone OK. Especially when regularizing on different things..
"""

spoke_result_dict = {'1kt_power': (DDATA_1KT_BETA_POWER, DPLOT_1KT_BETA_POWER),
                     '1kt_sar': (DDATA_1KT_BETA_VOP, DPLOT_1KT_BETA_VOP),
                     '5kt_power': (DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER),
                     '5kt_sar': (DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP)}

# Parameters...
Y_METRIC = 'peak_SAR'
X_METRIC = 'b1p_nrmse'
cpx_key = 'random_shim'
# icoil = 0
for icoil in range(6):
    selected_coil = COIL_NAME_ORDER[icoil]
    coil_plot_name = COIL_NAME_ORDER_TRANSLATOR[selected_coil]
    dpower = '/data/seb/paper/14T/plot_body_thomas_mask_rmse_power'
    dsar = '/data/seb/paper/14T/plot_body_thomas_mask_rmse_sar'
    # Create plot figures...
    n_models = len(COIL_NAME_ORDER)

    fig, ax = plt.subplots()
    visual_obj = VisualizeAllMetrics(ddest=dpower, opt_shim_str=f'opt_shim_00')
    visual_obj_sar = VisualizeAllMetrics(ddest=dsar, opt_shim_str=f'opt_shim_00')
    # SUBDIR_RANDOM_SHIM = 'random_shim'
    json_path = os.path.join(dpower, SUBDIR_RANDOM_SHIM, selected_coil, f'{cpx_key}.json')
    if os.path.isfile(json_path):
        result_dict = visual_obj._load_json(json_path, cpx_key=cpx_key)
        if X_METRIC in result_dict.keys():
            color = COLOR_DICT[selected_coil]
            x_metric_random = result_dict[X_METRIC]
            y_metric_random = np.array(result_dict[Y_METRIC]) * RF_SCALING_FACTOR_1KT ** 2
            x_metric_optim = visual_obj.optimized_json_data[selected_coil][X_METRIC]
            y_metric_optim = np.array(visual_obj.optimized_json_data[selected_coil][Y_METRIC]) * RF_SCALING_FACTOR_1KT ** 2
            x_metric_optim_sar = visual_obj_sar.optimized_json_data[selected_coil][X_METRIC]
            y_metric_optim_sar = np.array(
                visual_obj_sar.optimized_json_data[selected_coil][Y_METRIC]) * RF_SCALING_FACTOR_1KT ** 2
            ax.scatter(x_metric_random, y_metric_random, label='random', color=color, alpha=0.1)
            ax.scatter(x_metric_optim, y_metric_optim, label='opt power', color='r', alpha=0.5)
            ax.scatter(x_metric_optim_sar, y_metric_optim_sar, label='opt sar', color='b', alpha=0.5)

    counter = 0
    # Dont plot any 1kt stuff...
    for k, (ddata, dplot) in spoke_result_dict.items():
        if '1kt' in k:
            print(k)
            if counter == 0:
                sel_color = 'r'
            else:
                sel_color = 'b'
            kt_img = helper_14T.KtImage(ddata, dplot, selected_coil, weird_rf_factor=1, flip_angle_factor=TARGET_FLIP_ANGLE)
            temp_nrmse, temp_peak_sar = kt_img.get_nrmse_peak_sar()
            ax.scatter(temp_nrmse, temp_peak_sar, color=sel_color, label=k)
        else:
            continue
            print(k)
            if counter > 2:
                sel_color = 'g'
            else:
                sel_color = 'y'
            kt_img = helper_14T.KtImage(ddata, dplot, selected_coil, weird_rf_factor=WEIRD_RF_FACTOR, flip_angle_factor=TARGET_FLIP_ANGLE)
            temp_nrmse, temp_peak_sar = kt_img.get_nrmse_peak_sar()
            ax.scatter(temp_nrmse, temp_peak_sar, color=sel_color, label=k)
        #
        counter += 1

    ax.set_ylim(0, 2000)
    ax.set_xlim(0, 200)
    #
    plt.legend()
    fig.savefig(os.path.join(visual_obj.ddest, f'optimal_vs_random_spoke_{selected_coil}.png'), bbox_inches='tight')

    #
    # """
    # Inspect difference in both 1kt files... -> This is very little
    # """
    # ddata, dplot = (DDATA_1KT_BETA_POWER, DPLOT_1KT_BETA_POWER)
    # # ddata, dplot =  (DDATA_1KT_BETA_VOP, DPLOT_1KT_BETA_VOP)
    # # ddata, dplot = (DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER)
    #
    # # Lets inspect the peaksar calculated by Thomas...
    # kt_img = helper_14T.KtImage(ddata, dplot, selected_coil, weird_rf_factor=WEIRD_RF_FACTOR, flip_angle_factor=TARGET_FLIP_ANGLE)
    # result = []
    # for file_name in kt_img.output_design_files:
    #     sel_mat_file = os.path.join(kt_img.ddata, file_name)
    #     beta_value = kt_img._file_sorter(file_name)
    #     mat_obj = scipy.io.loadmat(sel_mat_file)
    #     peak_SAR = mat_obj['output']['rf'][0][0]['peakSAR'][0][0]
    #     rmse_value = mat_obj['output']['rf'][0][0]['rms'][0][0]
    #     solution = mat_obj['output']['RF_Waveforms_mT'][0][0].T
    #     result.append([peak_SAR, rmse_value, solution])
    #
    # res1, res2 = kt_img.get_nrmse_peak_sar()
    # # #
    # temp_peak_sar = []
    # temp_nrmse = []
    # for i_file in kt_img.output_design_files:
    #     rf_waveform = kt_img.get_unique_pulse_settings(i_file)
    #     # nc -> n pulses, c coils
    #     # cdz -> c coils, d coils, z number of VOPs
    #     # dn -> d coils, n pulses
    #     # Returns n pulses by z number of VOPs
    #     shimmed_VOP = np.einsum("nc,cdz,dn->nz", rf_waveform.T.conjugate(), kt_img.VOP_array, rf_waveform)
    #     # print('Hi, I am taking the mean over the number of pulses. Then take the maximum of the real part')
    #     # print(' I think this is wrong, but I am not sure. Looks weird to average something like peak SAR..')
    #     print(np.max(shimmed_VOP.mean(axis=0).real), np.max(shimmed_VOP.sum(axis=0).real))
    # # #
    # max(res2)
    # peaksar_list, _, _ = zip(*result)
    # print(peaksar_list[:10])
    # min(peaksar_list)
    # sum(peaksar_list) / len(peaksar_list)
    # max(peaksar_list)
    #
    # peaksar_list_copy = np.copy(peaksar_list)
    #
    # np.mean((np.array(peaksar_list) - peaksar_list_copy) ** 2 )
