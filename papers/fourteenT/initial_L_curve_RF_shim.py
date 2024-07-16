from objective_helper.fourteenT import VisualizeAllMetrics
import helper.misc as hmisc
import numpy as np
import matplotlib.pyplot as plt
import objective_helper.fourteenT as helper_14T
from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, COIL_NAME_ORDER, COLOR_DICT, \
    COIL_NAME_ORDER_TRANSLATOR, PLOT_LINEWIDTH, DPLOT_FINAL, OPTIMAL_SHIM_POWER, TARGET_FLIP_ANGLE, \
    OPTIMAL_SHIM_SAR, DPLOT_KT_BETA_POWER, DPLOT_KT_BETA_VOP, RF_SCALING_FACTOR
import os


"""
With this script we hope to plot the L curves and find the optimal points.
"""


def plot_all_results_single_figure(optimal_shim_results, fig=None, line_style='--', marker_style='o',
                                   key_y='peak_SAR', key_x='b1p_nrmse',
                                   xlim=100, ylim=50, label_appendix='',
                                   scaling_factor=1):
    if fig is None:
        fig, ax = plt.subplots(1, 2)
        ax = ax.ravel()
    else:
        ax = fig.get_axes()
#
    # Big plot to see all the simulation results
    for ii, sel_coil in enumerate(COIL_NAME_ORDER):
        print(sel_coil)
        coil_plot_name = COIL_NAME_ORDER_TRANSLATOR[sel_coil]
        all_coil_results = optimal_shim_results[sel_coil]
        all_coil_results_dict = hmisc.listdict2dictlist(all_coil_results)
        all_y_values = all_coil_results_dict[f'{key_y}']
        all_x_values = all_coil_results_dict[f'{key_x}']
        #
        x = np.array(all_x_values).ravel()
        y = np.array(all_y_values).ravel() * scaling_factor ** 2
        point_list = [(ix, iy) for ix, iy in list(zip(x, y))]
        lower_bound, index_lower_bound = hmisc.lower_bound_line(point_list)
        min_x_coords, min_y_coords = zip(*lower_bound)
        if ii == 0:
            ax[1].plot(min_x_coords, min_y_coords, color=COLOR_DICT[sel_coil], linestyle=line_style, label=label_appendix)
            # ax[1].plot(min_x_coords, min_y_coords, '-o', color=COLOR_DICT[sel_coil], label=label_appendix)
        else:
            # ax[1].plot(min_x_coords, min_y_coords, '-o', color=COLOR_DICT[sel_coil])
            ax[1].plot(min_x_coords, min_y_coords, color=COLOR_DICT[sel_coil], linestyle=line_style)
        #
        for jj in range(n_results):
            temp_all_y_values = np.array(all_y_values[jj]) * scaling_factor ** 2
            if jj == 0:
                ax[0].scatter(all_x_values[jj], temp_all_y_values, label=coil_plot_name + ' - ' + label_appendix,
                              color=COLOR_DICT[sel_coil], alpha=0.3, marker=marker_style)
            else:
                ax[0].scatter(all_x_values[jj], temp_all_y_values, color=COLOR_DICT[sel_coil],
                              alpha=0.3, marker=marker_style)
    ax[0].set_xlim(0, xlim)
    ax[0].set_ylim(0, ylim)
    ax[1].set_xlim(0, xlim)
    ax[1].set_ylim(0, ylim)
    return fig


def plot_all_results(optimal_shim_results, key_y='peak_SAR', key_x='b1p_nrmse', scaling_factor=1):
    fig, ax = plt.subplots(*hmisc.get_square(len(COIL_NAME_ORDER)))
    ax = ax.ravel()
    # Big plot to see all the simulation results
    for ii, sel_coil in enumerate(COIL_NAME_ORDER):
        coil_plot_name = COIL_NAME_ORDER_TRANSLATOR[sel_coil]
        all_coil_results = optimal_shim_results[sel_coil]
        all_coil_results_dict = hmisc.listdict2dictlist(all_coil_results)
        all_peak_SAR_values = all_coil_results_dict[f'{key_y}']
        all_peak_SAR_values = np.array(all_peak_SAR_values).ravel() * scaling_factor ** 2
        all_b1p_nrmse_values = all_coil_results_dict[f'{key_x}']
        for jj in range(n_results):
            ax[ii].scatter(all_b1p_nrmse_values[jj], all_peak_SAR_values[jj], label=jj)
        ax[ii].set_title(coil_plot_name)
        if (ii == 2) or (ii == 3):
            # Shrink current axis by 20%
            box = ax[ii].get_position()
            ax[ii].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax[ii].legend(loc='center right', bbox_to_anchor=(-0.1, 1))
    return fig


def plot_single_result(optimal_shim_results, sel_index, key_y='peak_SAR', key_x='b1p_nrmse', scaling_factor=1):
    """
    Here we can see which index we want to use...
    :param optimal_shim_results:
    :param sel_index:
    :return:
    """
    fig, ax = plt.subplots(*hmisc.get_square(len(COIL_NAME_ORDER)))
    ax = ax.ravel()
    # Big plot to see all the simulation results
    for ii, sel_coil in enumerate(COIL_NAME_ORDER):
        coil_plot_name = COIL_NAME_ORDER_TRANSLATOR[sel_coil]
        all_coil_results = optimal_shim_results[sel_coil]
        all_coil_results_dict = hmisc.listdict2dictlist(all_coil_results)
        all_peak_SAR_values = all_coil_results_dict[f'{key_y}'][sel_index]
        all_peak_SAR_values = np.array(all_peak_SAR_values).ravel() * scaling_factor ** 2
        all_b1p_nrmse_values = all_coil_results_dict[f'{key_x}'][sel_index]
        ax[ii].scatter(all_b1p_nrmse_values, all_peak_SAR_values, c='b', alpha=0.5)
        ax[ii].set_title(coil_plot_name)
        for jj in range(0, len(all_peak_SAR_values), 1):
            ax[ii].text(all_b1p_nrmse_values[jj], all_peak_SAR_values[jj], f'{str(jj)}')
    return fig

key_y = 'peak_SAR'
scaling_factor = RF_SCALING_FACTOR
ylim = 50  # For norm power
# ylim = 100 # for sar
for i_options in CALC_OPTIONS[0:1]:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    objective_str = i_options['objective_str']
#
    print(ddest)
    # Collect the minimum number of results frmo the coils
    n_results = min([len(os.listdir(os.path.join(ddest, 'optim_shim', x))) for x in COIL_NAME_ORDER])
    temp_list = []
    for ii in range(n_results):
        visual_obj = VisualizeAllMetrics(ddest=ddest, opt_shim_str=f'opt_shim_{str(ii).zfill(2)}')
        temp_list.append(visual_obj.optimized_json_data)
#
    optimal_shim_results = hmisc.listdict2dictlist(temp_list)
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    fig = plot_all_results_single_figure(optimal_shim_results, key_y=key_y, ylim=ylim, fig=fig,
                                         scaling_factor=scaling_factor, label_appendix='minimum line')
    ax_list = fig.get_axes()
    [x.legend() for x in ax_list]
    [x.grid() for x in ax_list]
    fig.savefig(os.path.join(ddest, f'L_curve_all_results_single_figure_{key_y}.png'))
    fig = plot_all_results(optimal_shim_results, key_y=key_y, scaling_factor=scaling_factor)
    fig.savefig(os.path.join(ddest, f'L_curve_all_results_{key_y}.png'))
    fig = plot_single_result(optimal_shim_results, sel_index=0, key_y=key_y, scaling_factor=scaling_factor)
    fig.savefig(os.path.join(ddest, f'L_curve_single_results_{key_y}.png'))
    plt.close('all')




# Try to plot both the results of Forward power regularization and VOP regularization in one figure
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax = ax.ravel()
line_style_list = ['--', '-']
marker_style_list = ['o', '<']
key_y = 'peak_SAR'
ylim = 10
for jj, i_options in enumerate(CALC_OPTIONS):
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    objective_str = i_options['objective_str']
    # Collect the minimum number of results frmo the coils
    n_results = min([len(os.listdir(os.path.join(ddest, 'optim_shim', x))) for x in COIL_NAME_ORDER])
    temp_list = []
    for ii in range(n_results):
        visual_obj = VisualizeAllMetrics(ddest=ddest, opt_shim_str=f'opt_shim_{str(ii).zfill(2)}')
        temp_list.append(visual_obj.optimized_json_data)
    optimal_shim_results = hmisc.listdict2dictlist(temp_list)
    legend_label = f'power'
    if 'sar' in ddest:
        legend_label = f'sar'
    fig = plot_all_results_single_figure(optimal_shim_results, fig=fig, line_style=line_style_list[jj],
                                         marker_style=marker_style_list[jj],
                                         key_y=key_y, ylim=ylim, label_appendix=legend_label)

ax_list = fig.get_axes()
[x.legend() for x in ax_list]
plt.grid()
fig.savefig(os.path.join(DPLOT_FINAL, 'compare_reg_SAR_and_fwd_power.png'))


# Now check if we can plot the relation between forward power and peak SAR for both regularization techniques
fig, ax = plt.subplots(1, 2)
ax = ax.ravel()
line_style_list = ['--', '-']
for jj, i_options in enumerate(CALC_OPTIONS):
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    objective_str = i_options['objective_str']
    # Collect the minimum number of results frmo the coils
    n_results = min([len(os.listdir(os.path.join(ddest, 'optim_shim', x))) for x in COIL_NAME_ORDER])
    temp_list = []
    for ii in range(n_results):
        visual_obj = VisualizeAllMetrics(ddest=ddest, opt_shim_str=f'opt_shim_{str(ii).zfill(2)}')
        temp_list.append(visual_obj.optimized_json_data)
    optimal_shim_results = hmisc.listdict2dictlist(temp_list)
    fig = plot_all_results_single_figure(optimal_shim_results, fig=fig,
                                         line_style=line_style_list[jj],
                                         key_y='norm_power')

for i_ax in fig.get_axes():
    i_ax.set_xlim(0, 200)
    i_ax.set_ylim(0, 200)

fig.savefig('/data/seb/test_norm_power.png')