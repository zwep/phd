import objective_helper.fourteenT as helper_14T
import helper.misc as hmisc

import matplotlib.pyplot as plt
from objective_configuration.fourteenT import COIL_NAME_ORDER, COLOR_DICT, \
    DPLOT_FINAL, \
    DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER, \
    DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP, \
    TARGET_FLIP_ANGLE, WEIRD_RF_FACTOR, \
    OPTIMAL_KT_POWER, OPTIMAL_KT_SAR,\
    DPLOT_1KT_BETA_POWER, DDATA_1KT_BETA_POWER, \
    DPLOT_1KT_BETA_VOP, DDATA_1KT_BETA_VOP
import os


def plot_kt_spoke_individual(ddata, dplot, step_size=1):
    """
    This plots the results in a different subplots for each coil

    :param ddata:
    :param dplot:
    :return:
    """
    n_models = len(COIL_NAME_ORDER)
    subplot_size = hmisc.get_square(n_models)
    fig, ax = plt.subplots(*subplot_size)
    fig.suptitle(os.path.basename(ddata))
    ax = ax.ravel()

    for icoil, sel_coil in enumerate(COIL_NAME_ORDER[-1:]):
        print('Plotting coil ', sel_coil)
        # The WEIRD_RF_FACTOR is only necessary for the 5kT data I think. Check with Thomas Roos?
        visual_obj = helper_14T.KtImage(ddata, dplot, sel_coil,
                                        weird_rf_factor=WEIRD_RF_FACTOR,
                                        flip_angle_factor=TARGET_FLIP_ANGLE)
        temp_nrmse, temp_peak_sar = visual_obj.get_nrmse_peak_sar_Q()
        ax[icoil].scatter(temp_nrmse, temp_peak_sar, color=COLOR_DICT[sel_coil], label=sel_coil)
        for ii in range(0, len(temp_nrmse), step_size):
           ax[icoil].text(temp_nrmse[ii], temp_peak_sar[ii] * 1.05, f'{ii}')
        ax[icoil].legend()
        ax[icoil].set_xlabel('NRMSE [%]')
        ax[icoil].set_ylabel('peak SAR (10g) [W/kg]')
#
    return fig


if __name__ == "__main__":
    five_spoke_results = [(DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER),  # Regularization on power
                          (DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP)]  # Regularization on SAR
#
    one_spoke_results = [(DDATA_1KT_BETA_POWER, DPLOT_1KT_BETA_POWER),  # Regularization on power
                         (DDATA_1KT_BETA_VOP, DPLOT_1KT_BETA_VOP)]  # Regularization on SAR
    #
    for i_ddata, i_plot in five_spoke_results:
        fig = plot_kt_spoke_individual(ddata=i_ddata, dplot=i_plot)
        fig.savefig(os.path.join(i_plot, 'L_curve_kT_spoke_indv.png'))
        plt.pause(0.1)

    for i_ddata, i_plot in one_spoke_results:
        fig = plot_kt_spoke_individual(ddata=i_ddata, dplot=i_plot)
        fig.savefig(os.path.join(i_plot, 'L_curve_kT_spoke_indv.png'))
        plt.pause(0.1)