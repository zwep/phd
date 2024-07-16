import objective_helper.fourteenT as helper_14T

import matplotlib.pyplot as plt
from objective_configuration.fourteenT import COIL_NAME_ORDER, COLOR_MAP, DPLOT_FINAL,\
    COIL_NAME_ORDER_TRANSLATOR, \
    DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER, TARGET_FLIP_ANGLE, \
    DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP, WEIRD_RF_FACTOR
import numpy as np
import os
import helper.plot_class as hplotc


"""
Plot the L curve when regularizared on FORWARD POWER
"""

n_coils = len(COIL_NAME_ORDER)

for sel_ddata, sel_dplot in [(DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER), (DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP)][0:1]:
    print(sel_ddata, sel_dplot)
    for icoil, sel_coil in enumerate(COIL_NAME_ORDER):
        print(icoil, sel_coil)
        visual_obj = helper_14T.VisualizeKtImage(sel_ddata, sel_dplot, sel_coil,
                                                 str_normalization='_head_sar',
                                                 weird_rf_factor=WEIRD_RF_FACTOR,
                                                 flip_angle_factor=TARGET_FLIP_ANGLE)
        multi_sar_distr = visual_obj.get_plot_sar_spokes()
        vmax = np.array(np.abs(multi_sar_distr)).max() * 0.80
        fig_obj_sar = hplotc.ListPlot([multi_sar_distr], augm='np.abs', cbar=False,
                                      cbar_round_n=0, wspace=0.0,
                                      ax_off=True, cmap=COLOR_MAP, vmin=(0, vmax))
#
        # Also plot the position with the highest peak SAR
        for i_img in range(len(multi_sar_distr)):
            max_index = np.argmax(multi_sar_distr[i_img])
            index_x, index_y = np.unravel_index(max_index, multi_sar_distr[i_img].shape)
            fig_obj_sar.ax_list[i_img].scatter(index_y, index_x, c='r', marker='*')
#
        subtitle = [f'kT spoke {i}' for i in range(1, len(multi_sar_distr))]
        subtitle[0] = COIL_NAME_ORDER_TRANSLATOR[sel_coil] + '\n' + subtitle[0]
        subtitle = subtitle + ['Averaged']
#
        fig_obj_sar_cbar = hplotc.ListPlot([multi_sar_distr], augm='np.abs', cbar=True,
                                           cbar_round_n=2, wspace=0.0,
                                           ax_off=True, cmap=COLOR_MAP,
                                           subtitle=[subtitle], vmin=(0, vmax))
        # Also plot the position with the highest peak SAR
        for i_img in range(len(multi_sar_distr)):
            max_index = np.argmax(multi_sar_distr[i_img])
            index_x, index_y = np.unravel_index(max_index, multi_sar_distr[i_img].shape)
            fig_obj_sar_cbar.ax_list[i_img].scatter(index_y, index_x, c='r', marker='*')
#
        fig_obj_sar.figure.savefig(os.path.join(sel_dplot, f'SAR_plot_kt_spokes_{sel_coil}.png'),
                                   bbox_inches='tight',
                                   pad_inches=0)
        fig_obj_sar_cbar.figure.savefig(os.path.join(sel_dplot, f'cbar_SAR_plot_kt_spokes_{sel_coil}.png'), bbox_inches='tight')
        hplotc.close_all()
