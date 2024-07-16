import objective_helper.fourteenT as helper_14T
import helper.misc as hmisc
import matplotlib.pyplot as plt
from objective_configuration.fourteenT import COIL_NAME_ORDER, COLOR_DICT, \
    DPLOT_FINAL, \
    DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER, \
    DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP, \
    TARGET_FLIP_ANGLE, WEIRD_RF_FACTOR, \
    OPTIMAL_KT_POWER, OPTIMAL_KT_SAR,\
    OPTIMAL_KT_POWER_1kt, OPTIMAL_KT_SAR_1kt,\
    DPLOT_1KT_BETA_POWER, DDATA_1KT_BETA_POWER, \
    DPLOT_1KT_BETA_VOP, DDATA_1KT_BETA_VOP,\
    DPLOT_FINAL
import os

"""
Here we plot each 
"""



def get_plot_kt_spoke_coil(sel_coil, ddata, dplot):
    """
    This plots the results of the kT point/spokes method in a single graph
    for all the coil designs that we tested.
    No scaling is done on the result

    :param ddata:
    :param dplot:
    :return:
    """
    visual_obj = helper_14T.KtImage(ddata, dplot, sel_coil,
                                    weird_rf_factor=WEIRD_RF_FACTOR,
                                    flip_angle_factor=TARGET_FLIP_ANGLE)
#
    temp_nrmse, temp_peak_sar = visual_obj.get_nrmse_peak_sar()
    return temp_nrmse, temp_peak_sar


def plot_kt_spoke(ddata, dplot, optim_shim_index, ax=None, marker='o', **kwargs):
    """
    This plots the results of the kT point/spokes method in a single graph
    for all the coil designs that we tested.
    No scaling is done on the result

    :param ddata:
    :param dplot:
    :return:
    """
    for icoil, sel_coil in enumerate(COIL_NAME_ORDER):
        temp_nrmse, temp_peak_sar = get_plot_kt_spoke_coil(sel_coil, ddata, dplot)
        ax.scatter(temp_nrmse, temp_peak_sar, color=COLOR_DICT[sel_coil], label=sel_coil, marker=marker, **kwargs)
        index_stuff = optim_shim_index[sel_coil]
        ax.scatter(temp_nrmse[index_stuff], temp_peak_sar[index_stuff], marker='*', c='k', zorder=999)
#
    ax.legend()
    ax.set_xlabel('NRMSE [%]')
    ax.set_ylabel('peak SAR (10g) [W/kg]')
    return ax


def select_optimal_dict(ddata):
    if 'power' in ddata:
        if '1kt' in ddata:
            optim_shim_index = OPTIMAL_KT_POWER_1kt
        else:
            optim_shim_index = OPTIMAL_KT_POWER
    else:
        if '1kt' in ddata:
            optim_shim_index = OPTIMAL_KT_SAR_1kt
        else:
            optim_shim_index = OPTIMAL_KT_SAR
#
    return optim_shim_index


if __name__ == "__main__":
    five_spoke_results = [(DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER),  # Regularization on power
                          (DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP)]  # Regularization on SAR
#
    one_spoke_results = [(DDATA_1KT_BETA_POWER, DPLOT_1KT_BETA_POWER),  # Regularization on power
                         (DDATA_1KT_BETA_VOP, DPLOT_1KT_BETA_VOP)]  # Regularization on SAR

    # Plot the L curves from both optimizations in one graph
    fig, ax = plt.subplots(2, figsize=(15, 10))
    for ii, (i_ddata, i_plot) in enumerate(five_spoke_results[0:1]):
        optim_shim_index = select_optimal_dict(i_ddata)
        ax[ii] = plot_kt_spoke(ddata=i_ddata, dplot=i_plot, optim_shim_index=optim_shim_index,
                               ax=ax[ii])
        ax[ii].set_xlim(0, 20)
        ax[ii].set_ylim(0, 500)
        plt.pause(0.1)

    fig.savefig(os.path.join(DPLOT_FINAL, 'L_curve_kT_spoke_power_and_SAR_side_by_side.png'), bbox_inches='tight', pad_inches=0)

    # Plot the L curves from both optimizations in one graph
    fig, ax = plt.subplots(figsize=(15, 10))
    marker_list = ['o', '>']
    for ii, (i_ddata, i_plot) in enumerate(five_spoke_results):
        optim_shim_index = select_optimal_dict(i_ddata)
        ax = plot_kt_spoke(ddata=i_ddata, dplot=i_plot, optim_shim_index=optim_shim_index, ax=ax,
                           marker=marker_list[ii], facecolor='none')
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 500)

    fig.savefig(os.path.join(DPLOT_FINAL, 'L_curve_kT_spoke_power_and_SAR_single_figure.png'), bbox_inches='tight', pad_inches=0)

    # Plot each L curve of each individual optimization in one graph
    for ii, (i_ddata, i_plot) in enumerate(five_spoke_results):
        fig, ax = plt.subplots()
        optim_shim_index = select_optimal_dict(i_ddata)
        ax = plot_kt_spoke(ddata=i_ddata, dplot=i_plot, optim_shim_index=optim_shim_index, ax=ax)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 500)
        fig.savefig(os.path.join(i_plot, 'L_curve_kT_spoke.png'))

    i_ddata, i_plot = five_spoke_results[0]
    a,b=get_plot_kt_spoke_coil(sel_coil='8 Channel Loop Array big', ddata=i_ddata, dplot=i_plot)
    a[16]
    b[16]
    """
    Do the same for the 1kT simulations
    """

    fig, ax = plt.subplots(2)
    for ii, (i_ddata, i_plot) in enumerate(one_spoke_results):
        optim_shim_index = select_optimal_dict(i_ddata)
        ax[ii] = plot_kt_spoke(ddata=i_ddata, dplot=i_plot, optim_shim_index=optim_shim_index,
                               ax=ax[ii])
    [x.set_xlim(0, 100) for x in ax]
    # [x.set_ylim(0, 1e6) for x in ax]
    fig.savefig(os.path.join(DPLOT_FINAL, 'L_curve_1kT_spoke_power_and_SAR.png'))

    # Plot each L curve of each individual optimization in one graph
    for ii, (i_ddata, i_plot) in enumerate(one_spoke_results):
        fig, ax = plt.subplots()
        optim_shim_index = select_optimal_dict(i_ddata)
        ax = plot_kt_spoke(ddata=i_ddata, dplot=i_plot, optim_shim_index=optim_shim_index, ax=ax)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 500)
        fig.savefig(os.path.join(i_plot, 'L_curve_kT_spoke.png'))

