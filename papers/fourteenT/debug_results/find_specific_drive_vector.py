from objective_helper.fourteenT import VisualizeAllMetrics, DataCollector
from objective_helper.fourteenT import VisualizeAllMetrics, DataCollector, ReadMatData
import matplotlib.pyplot as plt
import scipy.io
import helper.misc as hmisc
import os
import objective_helper.fourteenT as helper_14T
from objective_configuration.fourteenT import RF_SCALING_FACTOR, COLOR_MAP, MID_SLICE_OFFSET, COIL_NAME_ORDER, CALC_OPTIONS, DDATA, COLOR_DICT, \
    DPLOT_1KT_BETA_POWER, DDATA_1KT_BETA_POWER, \
    DPLOT_1KT_BETA_VOP, DDATA_1KT_BETA_VOP, \
    DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP, \
    DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER, \
    COIL_NAME_ORDER_TRANSLATOR, RF_SCALING_FACTOR_1KT, RF_SCALING_FACTOR, \
    RF_SCALING_FACTOR, WEIRD_RF_FACTOR, TARGET_FLIP_ANGLE, SUBDIR_RANDOM_SHIM
import os
import numpy as np

"""
Find a drive vector based on the L curves from the optimal stuff


We found out that the calculations were wrong,,, somehow???
This is messed up

"""

file_list = os.listdir(DDATA)
mat_files = [x for x in file_list if x.endswith('mat')]
i_options_sar = [x for x in CALC_OPTIONS if 'power' in x['ddest']][0]
full_mask = i_options_sar['full_mask']
type_mask = i_options_sar['type_mask']
ddest = i_options_sar['ddest']
objective_str = i_options_sar['objective_str']
# Collect the minimum number of results frmo the coils
n_results = min([len(os.listdir(os.path.join(ddest, 'optim_shim_recalc_sar', x))) for x in COIL_NAME_ORDER])
temp_list = []
for ii in range(n_results):
    visual_obj = VisualizeAllMetrics(ddest=ddest, opt_shim_str=f'opt_shim_{str(ii).zfill(2)}')
    temp_list.append(visual_obj.optimized_json_data)


optimal_shim_results = hmisc.listdict2dictlist(temp_list)


# Check the calculated values
for sel_coil_key, v in COIL_NAME_ORDER_TRANSLATOR.items():
    print(sel_coil_key)
    sel_mat_file = sel_coil_key + '_ProcessedData.mat'
    result_list = optimal_shim_results[sel_coil_key]
    optim_index = 10
    lambda_index = 10
    opt_shim = result_list[optim_index]['opt_shim'][lambda_index]
    peak_SAR = result_list[optim_index]['peak_SAR'][lambda_index]
    peak_SAR_40 = result_list[optim_index]['peak_SAR'][lambda_index]
    print('Previous found VOP ', peak_SAR_40)
    mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
    data_obj = DataCollector(mat_reader, full_mask=full_mask, type_mask=type_mask)
    shimmed_b1 = data_obj.get_shimmed_b1p(opt_shim)
    peakSAR, peakSAR_normalized, optm_power_deposition = data_obj.get_peak_SAR_VOP(opt_shim)
    print('newly calculated VOP', peakSAR)
    #

# Now for random shim..

# Check the calculated values
for sel_coil_key, v in COIL_NAME_ORDER_TRANSLATOR.items():
    print(sel_coil_key)
    sel_mat_file = sel_coil_key + '_ProcessedData.mat'
    mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
    data_obj = DataCollector(mat_reader, full_mask=full_mask, type_mask=type_mask)
    visual_obj = VisualizeAllMetrics(ddest=ddest, opt_shim_str=f'opt_shim_00')
    SUBDIR_RANDOM_SHIM = 'random_shim'
    cpx_key = 'random_shim'
    json_path = os.path.join(ddest, SUBDIR_RANDOM_SHIM, sel_coil_key, f'{cpx_key}.json')
    result_dict = visual_obj._load_json(json_path, cpx_key=cpx_key)
    n_shim = len(result_dict['random_shim'])
    sel_shim = np.random.randint(0, n_shim, 10)
    for ii in sel_shim:
        temp_shim = result_dict['random_shim'][ii]
        temp_peakSAR = result_dict['peak_SAR'][ii]
        peakSAR, peakSAR_normalized, optm_power_deposition = data_obj.get_peak_SAR_VOP(temp_shim)
        print(np.round(peakSAR, 2), np.round(temp_peakSAR, 2), np.round(peakSAR / temp_peakSAR, 2))


"""
below we tried to find the distributions of sar and b1 for the 'optimal' shim settings
"""


def get_optimal_index(result_list, x_key='b1p_nrmse', y_key='peak_SAR',  lower_y=0., upper_y=10., lower_x=10., upper_x=60.,
                      scaling_factor=1):
    # With this we can get a specific shim setting under constrains for the x and y axis
    # This iwas initially used to check the distributions
    # But we noticed that after recalculation we have made some mistakes for the SAR optimized values..?
    index_result = []
    # Returns each 'optimal' or minimal point over the simulations...
    for ii, i_result_set in enumerate(result_list):
        x_values = np.array(i_result_set[x_key])
        y_values = np.array(i_result_set[y_key]) * scaling_factor ** 2
        x_bin = (lower_x < x_values) * (x_values < upper_x)
        y_bin = (lower_y < y_values) * (y_values < upper_y)
        if any(x_bin * y_bin):
            # print(ii, np.argwhere(x_bin * y_bin))
            found_indices = list(np.argwhere(x_bin * y_bin).ravel())
            selected_minimum_index = found_indices[np.argmin(y_values[found_indices])]
            index_result.append((min(y_values[found_indices]), ii, selected_minimum_index))
    index_result = sorted(index_result, key=lambda x: x[0])
    return index_result

from objective_helper.fourteenT import VisualizeAllMetrics, DataCollector, ReadMatData
import helper.plot_fun as hplotf
import helper.plot_class as hplotc

sel_coil_abrev = '8L'
sel_coil_key = [k for k, v in COIL_NAME_ORDER_TRANSLATOR.items() if sel_coil_abrev == v][0]
result_list = optimal_shim_results[sel_coil_key]
search_result = get_optimal_index(result_list,
                                  lower_y=25,
                                  upper_y=30,
                                  lower_x=65,
                                  upper_x=75,
                                  scaling_factor=RF_SCALING_FACTOR)
print(search_result)
_, optim_index, lambda_index = search_result[0]
print(sel_coil_key, optim_index, lambda_index)
# opt_shim = result_list[optim_index]['opt_shim'][lambda_index]
# print(np.linalg.norm(np.abs(opt_shim)))
# result_list[optim_index]['peak_SAR'][lambda_index]
# result_list[optim_index]['b1p_nrmse'][lambda_index]
#
#
# # Now get the mat files again....
# sel_mat_file = sel_coil_key + '_ProcessedData.mat'
#
# mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
# data_obj = DataCollector(mat_reader, full_mask=full_mask, type_mask=type_mask)
# opt_shim = result_list[optim_index]['opt_shim'][lambda_index]
# rmse, nrmse, avg_b1, rmse_normalized, nrmse_normalized, avg_b1_normalized = data_obj.get_nrmse_b1p(x_shim=opt_shim, x_target=1)
# peakSAR, peakSAR_normalized, _ = data_obj.get_peak_SAR_VOP(opt_shim * 1)
# print(peakSAR)
#
# shimmed_b1, correction_head_sar, _ = data_obj.get_shimmed_b1p(opt_shim)
#
# b1_plot_list = hplotf.get_all_mid_slices(shimmed_b1[::-1], offset=MID_SLICE_OFFSET)
# fig_obj = hplotc.ListPlot([b1_plot_list], ax_off=True, cbar=True, wspace=0.5, title=mat_reader.coil_name, cmap=COLOR_MAP)
# fig_obj.figure.savefig(os.path.join(ddest, f'{mat_reader.coil_name}_b1file.png'))
# # Same thing for SAR...
# Q_container = mat_reader.read_Q_object()
# sar_shimmed = np.einsum("d, dczyx, c -> zyx", opt_shim.conjugate(), Q_container['Q10g'], opt_shim)
#
# b1_plot_list = hplotf.get_all_mid_slices(sar_shimmed.real[::-1], offset=MID_SLICE_OFFSET)
# fig_obj = hplotc.ListPlot([b1_plot_list], ax_off=True, cbar=True, wspace=0.5, title=mat_reader.coil_name, cmap=COLOR_MAP)
# fig_obj.figure.savefig(os.path.join(ddest, f'{mat_reader.coil_name}_sarfile.png'))


"""
DO the same for all the other coils
"""

# sel_coil_abrev = '8D8L'
# sel_coil_abrev = '15D'
# # sel_coil_abrev = '8D8L'
# for sel_coil_abrev in ['8L']:
for sel_coil_key in COIL_NAME_ORDER_TRANSLATOR.keys():
    # sel_coil_key = [k for k, v in COIL_NAME_ORDER_TRANSLATOR.items() if sel_coil_abrev == v][0]
    sel_mat_file = sel_coil_key + '_ProcessedData.mat'
    print(sel_mat_file)
    result_list = optimal_shim_results[sel_coil_key]
    _, optim_index, lambda_index = get_optimal_index(result_list, upper_x=5, lower_y=30, upper_y=40)[0]
    opt_shim = result_list[optim_index]['opt_shim'][lambda_index]
    #
    # Now get the mat files again....
    mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
    data_obj = DataCollector(mat_reader, full_mask=full_mask, type_mask=type_mask)
    rmse, nrmse, avg_b1, rmse_normalized, nrmse_normalized, avg_b1_normalized = data_obj.get_nrmse_b1p(x_shim=opt_shim, x_target=1)
    peakSAR, peakSAR_normalized, _ = data_obj.get_peak_SAR_VOP(opt_shim * 1)
    shimmed_b1, correction_head_sar, _ = data_obj.get_shimmed_b1p(opt_shim)
    #
    # #
    b1_plot_list = hplotf.get_all_mid_slices(shimmed_b1[::-1], offset=MID_SLICE_OFFSET)
    fig_obj = hplotc.ListPlot([b1_plot_list], ax_off=True, cbar=True, wspace=0.5, title=mat_reader.coil_name, cmap=COLOR_MAP)
    fig_obj.figure.savefig(os.path.join(ddest, f'{mat_reader.coil_name}_b1file.png'))
    # Same thing for SAR...
    Q_container = mat_reader.read_Q_object()
    sar_shimmed = np.einsum("d, dczyx, c -> zyx", opt_shim.conjugate(), Q_container['Q10g'], opt_shim)
    print('Calculated VOP ', peakSAR)
    print('max SAR based on real Q10g', np.max(sar_shimmed.real * data_obj.selected_mask))

b1_plot_list = hplotf.get_all_mid_slices(sar_shimmed.real[::-1], offset=MID_SLICE_OFFSET)
fig_obj = hplotc.ListPlot([b1_plot_list], ax_off=True, cbar=True, wspace=0.5, title=mat_reader.coil_name, cmap=COLOR_MAP)
fig_obj.figure.savefig(os.path.join(ddest, f'{mat_reader.coil_name}_sarfile.png'))
