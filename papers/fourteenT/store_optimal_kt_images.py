import pandas as pd
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import numpy as np
import objective_helper.fourteenT as helper_14T
from objective_configuration.fourteenT import COIL_NAME_ORDER, \
    DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP,\
    DPLOT_KT_BETA_POWER, DDATA_KT_BETA_POWER,\
    DDATA_KT_POWER, DPLOT_KT_POWER, \
    TARGET_FLIP_ANGLE, WEIRD_RF_FACTOR, OPTIMAL_KT_POWER, \
    DDATA_KT_VOP, DPLOT_KT_VOP, OPTIMAL_KT_SAR, MID_SLICE_OFFSET, COLOR_MAP


"""
DDATA_KT_BETA_POWER contains the kT solutions to the minimization problem regularized with forward power

Here we 
- Store the result frm the "optimal" solutions. These values were chosen from the L-curve plot
- Report on the rf norm and head sar per coil
"""


sel_kt_spoke = 5
head_sar_dict = {}
for icoil, sel_coil in enumerate(COIL_NAME_ORDER):
    print(sel_coil)
    _ = head_sar_dict.setdefault(sel_coil, {})
    visual_obj = helper_14T.StoreKtImage(DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER, sel_coil,
                                         weird_rf_factor=WEIRD_RF_FACTOR,
                                         flip_angle_factor=TARGET_FLIP_ANGLE,
                                         load_Q=False)
    # Select the file that is optimal in a sense..
    file_name = visual_obj.output_design_files[OPTIMAL_KT_POWER[sel_coil]]
    # SH sept 2023: adding manual plotting of the flip angle map
    # fa_array = visual_obj.get_flip_angle_map(file_name)
    # fa_array_list = hplotf.get_all_mid_slices(fa_array[::-1], offset=MID_SLICE_OFFSET)
    # fig_obj = hplotc.ListPlot([fa_array_list], ax_off=True, cbar=True, wspace=0.5, vmin=(0, 40),
    #                           title=sel_coil, cmap=COLOR_MAP)
    # fig_obj.figure.savefig(visual_obj.path_flip_angle_file + "_" + sel_coil + '.png', bbox_inches='tight')
    # fig_obj.figure.savefig(visual_obj.path_flip_angle_file + "_" + sel_coil + '.svg', bbox_inches='tight')

    visual_obj.store_flip_angle(file_name)
    # max_peak_sar_Q, _, _ = visual_obj.get_peak_SAR_Q(file_name)
    # max_peak_sar, _, _ = visual_obj.get_peak_SAR(file_name)
    # print(max_peak_sar, max_peak_sar_Q)
    avg_power_deposition = visual_obj.store_time_avg_SAR(file_name)
    norm_avg_rf_waveform = np.sum(np.mean(np.abs(visual_obj.get_unique_pulse_settings(file_name)), axis=1) ** 2, axis=0)
    head_sar_dict[sel_coil]['rf_norm'] = norm_avg_rf_waveform
    head_sar_dict[sel_coil]['head_sar'] = avg_power_deposition / sel_kt_spoke
    del visual_obj

rf_waveform = visual_obj.get_unique_pulse_settings(file_name)
z = visual_obj.mat_reader.read_B1_object()
avg_B1 = np.einsum("nc,czyx->nzyx", rf_waveform.T, z['b1p'])
np.abs(avg_B1[:, visual_obj.thomas_mask_array==1].sum(axis=0)).mean()
np.abs(avg_B1[:, visual_obj.thomas_mask_array==1]).sum(axis=0).mean()

avg_B1_sel_x, avg_B1_sel_y, avg_B1_sel_z = zip(*[hplotf.get_all_mid_slices(x, offset=MID_SLICE_OFFSET) for x in avg_B1])
hplotc.ListPlot(np.sum(np.array(avg_B1_sel_x)[..., ::-1, :], axis=0))

import pandas as pd
pd_frame = pd.DataFrame(head_sar_dict)
print((pd_frame[COIL_NAME_ORDER]).T.round(2))#.to_csv(sep='\t'))

"""
DDATA_KT_BETA_VOP contains the kT solutions to the minimization problem regularized with peak SAR

Here we 
- Store the result frm the "optimal" solutions. These values were chosen from the L-curve plot
- Report on the rf norm and head sar per coil
"""

#
# head_sar_dict = {}
# for icoil, sel_coil in enumerate(COIL_NAME_ORDER):
#     print(sel_coil)
#     _ = head_sar_dict.setdefault(sel_coil, {})
#     visual_obj = helper_14T.StoreKtImage(DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP, sel_coil)
#     file_name = visual_obj.output_design_files[OPTIMAL_KT_SAR[sel_coil]]
#     visual_obj.store_flip_angle(file_name)
#     avg_power_deposition = visual_obj.store_time_avg_SAR(file_name)
#     norm_avg_rf_waveform = np.sum(np.mean(np.abs(visual_obj.get_unique_pulse_settings(file_name)), axis=1) ** 2, axis=0)
#     head_sar_dict[sel_coil]['rf_norm'] = norm_avg_rf_waveform
#     head_sar_dict[sel_coil]['head_sar'] = avg_power_deposition / sel_kt_spoke
#     del visual_obj
#
# import pandas as pd
# pd_frame = pd.DataFrame(head_sar_dict)
# print((pd_frame[COIL_NAME_ORDER]).round(2).to_csv(sep='\t'))