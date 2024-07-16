import pandas as pd
import numpy as np
import objective_helper.fourteenT as helper_14T
from objective_configuration.fourteenT import COIL_NAME_ORDER, \
    DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP,\
    DPLOT_KT_BETA_POWER, DDATA_KT_BETA_POWER,\
    DPLOT_1KT_BETA_POWER, DDATA_1KT_BETA_POWER,\
    DDATA_KT_POWER, DPLOT_KT_POWER, \
    TARGET_FLIP_ANGLE, WEIRD_RF_FACTOR, OPTIMAL_KT_POWER, \
    DDATA_KT_VOP, DPLOT_KT_VOP, OPTIMAL_KT_SAR

from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, RF_SCALING_FACTOR
from objective_helper.fourteenT import ReadMatData, StoreOptimizeData
import os



"""
For the kT stuff results..
"""

# Calculate difference for 5kT spokes sim
for icoil, sel_coil in enumerate(COIL_NAME_ORDER):
    print(sel_coil)
    visual_obj = helper_14T.StoreKtImage(DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER, sel_coil, weird_rf_factor=WEIRD_RF_FACTOR,
                                         flip_angle_factor=TARGET_FLIP_ANGLE)
    # Select the file that is optimal in a sense..
    for file_name in visual_obj.output_design_files:
        # rf_waveform = visual_obj.get_unique_pulse_settings(file_name)
        max_peak_sar_Q, _, _ = visual_obj.get_peak_SAR_Q(file_name)
        max_peak_sar, _, _ = visual_obj.get_peak_SAR(file_name)
        print(max_peak_sar, max_peak_sar_Q)


# Calculate differece for 1kT spokes sim
for icoil, sel_coil in enumerate(COIL_NAME_ORDER):
    print(sel_coil)
    visual_obj = helper_14T.StoreKtImage(DDATA_1KT_BETA_POWER, DPLOT_1KT_BETA_POWER, sel_coil,
                                         weird_rf_factor=WEIRD_RF_FACTOR,
                                         flip_angle_factor=TARGET_FLIP_ANGLE)
    # Select the file that is optimal in a sense..
    for file_name in visual_obj.output_design_files:
        max_peak_sar_Q, _, _ = visual_obj.get_peak_SAR_Q(file_name)
        max_peak_sar, _, _ = visual_obj.get_peak_SAR(file_name)
        print(max_peak_sar, max_peak_sar_Q)

"""
For the RF shims results..
"""

for i_options in CALC_OPTIONS[0:1]:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    # Doing so will leave us with a single instance of VisualizeData / ReadMatData
    visual_obj = None
    mat_reader = None
    for sel_coil in COIL_NAME_ORDER:
        print(sel_coil)
        sel_mat_file = sel_coil + "_ProcessedData.mat"
        mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
        del visual_obj
        del mat_reader
        mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
        # SO here I need to enter the optimal file string....
        visual_obj = StoreOptimizeData(ddest=ddest, mat_reader=mat_reader,
                                       full_mask=full_mask, type_mask=type_mask)
        for i_shim in visual_obj.result_dict['opt_shim'][:2]:
            peaksar_vop, _, _ = visual_obj.get_peak_SAR_VOP(i_shim)
            peaksar_vop = np.round(peaksar_vop, 2)
            sar_shimmed, _, _ = visual_obj.get_shimmed_sar_q(i_shim)
            peaksar_q = np.max(sar_shimmed.real)
            peaksar_q = np.round(peaksar_q, 2)
            print(peaksar_vop, peaksar_q, peaksar_vop / peaksar_q, np.linalg.norm(i_shim))