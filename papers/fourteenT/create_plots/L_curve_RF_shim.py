from objective_helper.fourteenT import VisualizeAllMetrics
import helper.misc as hmisc
import numpy as np
import matplotlib.pyplot as plt
import objective_helper.fourteenT as helper_14T
from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, COIL_NAME_ORDER, COLOR_DICT, \
    COIL_NAME_ORDER_TRANSLATOR, PLOT_LINEWIDTH, DPLOT_FINAL, OPTIMAL_SHIM_POWER, TARGET_FLIP_ANGLE, \
    OPTIMAL_SHIM_SAR, DPLOT_KT_BETA_POWER, DPLOT_KT_BETA_VOP, RF_SCALING_FACTOR
import os


def plot_final_result(optimal_shim_results, optimal_indices_dict, scaling_factor=1,
                      line_style='-', key_x='b1p_nrmse', key_y='peak_SAR'):
    """
    This one plots the final image we use in the paper
    :param optimal_shim_results:
    :param sel_index: the selected optima lshim results
    :return:
    """
#
    fig, ax = plt.subplots()
    for ii, sel_coil in enumerate(COIL_NAME_ORDER):
        coil_plot_name = COIL_NAME_ORDER_TRANSLATOR[sel_coil]
        sel_index_opt, sel_index_lambda = optimal_indices_dict[sel_coil]
        all_coil_results = optimal_shim_results[sel_coil]
        all_coil_results_dict = hmisc.listdict2dictlist(all_coil_results)
        chosen_y_value = all_coil_results_dict[key_y][sel_index_opt][sel_index_lambda] * scaling_factor ** 2
        chosen_x_value = all_coil_results_dict[key_x][sel_index_opt][sel_index_lambda]
        # Trying to fit in the minimum L curve line..!!!
        x = np.array(all_coil_results_dict[key_x]).ravel()
        y = np.array(all_coil_results_dict[key_y]).ravel() * scaling_factor ** 2
        point_list = [(ix, iy) for ix, iy in list(zip(x, y))]
        lower_bound, index_lower_bound = hmisc.lower_bound_line(point_list)
        min_x_coords, min_y_coords = zip(*lower_bound)
        # Plot the minimum line
        ax.plot(min_x_coords, min_y_coords, color=COLOR_DICT[sel_coil], linestyle=line_style, label=coil_plot_name,
                linewidth=PLOT_LINEWIDTH)
        #ax.scatter(min_x_coords, min_y_coords, marker='*', c='k')
        # Plot the black star//
        ax.scatter(chosen_x_value, chosen_y_value, marker='*', c='k', zorder=999)
        print(sel_coil, chosen_x_value, chosen_y_value, index_lower_bound)
#
    legend_obj = ax.legend(loc='upper right')
    # helper_14T.flush_right_legend(legend_obj)
    ax.set_xlabel('NRMSE [%]')
    ax.set_ylabel('peak SAR (10g) [W/kg]')
    return fig


for i_options in CALC_OPTIONS[:1]:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    objective_str = i_options['objective_str']
#
    # Collect the minimum number of results frmo the coils
    n_results = min([len(os.listdir(os.path.join(ddest, 'optim_shim', x))) for x in COIL_NAME_ORDER])
    temp_list = []
    for ii in range(n_results):
        visual_obj = VisualizeAllMetrics(ddest=ddest, opt_shim_str=f'opt_shim_{str(ii).zfill(2)}')
        temp_list.append(visual_obj.optimized_json_data)
#
    optimal_shim_results = hmisc.listdict2dictlist(temp_list)
#
    if 'power' in objective_str:
        fig = plot_final_result(optimal_shim_results, optimal_indices_dict=OPTIMAL_SHIM_POWER, scaling_factor=RF_SCALING_FACTOR)
        ax = fig.get_axes()[0]
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 50)
    else:
        fig = plot_final_result(optimal_shim_results, optimal_indices_dict=OPTIMAL_SHIM_SAR, scaling_factor=RF_SCALING_FACTOR)
        ax = fig.get_axes()[0]
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
#
    fig.savefig(os.path.join(ddest, 'L_curve_RF_shim.png'))
