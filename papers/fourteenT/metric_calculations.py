import objective_helper.fourteenT as helper_14T

import matplotlib.pyplot as plt
from objective_configuration.fourteenT import COIL_NAME_ORDER, COLOR_MAP, COLOR_DICT, \
    DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP, \
    DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER, \
    DDATA_KT_POWER, DPLOT_KT_POWER, WEIRD_RF_FACTOR, DDATA_1KT_BETA_POWER, DPLOT_1KT_BETA_POWER, TARGET_FLIP_ANGLE
import os
import helper.plot_class as hplotc
import pandas as pd
import numpy as np
from objective_configuration.fourteenT import COIL_NAME_ORDER, \
    DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP,\
    DPLOT_KT_BETA_POWER, DDATA_KT_BETA_POWER,\
    DPLOT_1KT_BETA_POWER, DDATA_1KT_BETA_POWER,\
    DDATA_KT_POWER, DPLOT_KT_POWER, \
    TARGET_FLIP_ANGLE, WEIRD_RF_FACTOR, OPTIMAL_KT_POWER, \
    DDATA_KT_VOP, DPLOT_KT_VOP, OPTIMAL_KT_SAR

from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, RF_SCALING_FACTOR
from objective_helper.fourteenT import ReadMatData, StoreOptimizeData, OptimizeData
import os


"""
The metric calculation done here is totally dependent on the file `store_kt_images.py`...
"""

metric_data_frame = None
for icoil, sel_coil in enumerate(COIL_NAME_ORDER):
    visual_obj = helper_14T.VisualizeKtImage(DPLOT_KT_BETA_POWER, DPLOT_KT_BETA_POWER, sel_coil, str_normalization='_forward_power',
                                             flip_angle_factor=TARGET_FLIP_ANGLE, weird_rf_factor=WEIRD_RF_FACTOR)
    metric_data_frame = visual_obj.report_metrics(metric_data_frame)
    del visual_obj

print('Metrics from KT BETA POWER')
print(metric_data_frame.T.round(4).to_csv(sep='\t   \t'))
print(metric_data_frame.T.round(4))  #.to_csv(sep='\t   \t'))

metric_data_frame = None
for icoil, sel_coil in enumerate(COIL_NAME_ORDER):
    visual_obj = helper_14T.VisualizeKtImage(DPLOT_KT_BETA_POWER, DPLOT_KT_BETA_POWER, sel_coil, str_normalization='_head_sar',
                                             flip_angle_factor=TARGET_FLIP_ANGLE, weird_rf_factor=WEIRD_RF_FACTOR)
    metric_data_frame = visual_obj.report_metrics(metric_data_frame)
    del visual_obj

print('Metrics from KT BETA POWER - headsar')
print(metric_data_frame.T.round(4).to_csv(sep='\t'))


"""
Here we calculate the metrics with the RF shims...
"""

pd_dataframe = None
for i_options in CALC_OPTIONS[0:1]:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    objective_str = i_options['objective_str']
    # Doing so will leave us with a single instance of VisualizeData / ReadMatData
    visual_obj = None
    mat_reader = None
    print(ddest)
    for sel_coil in COIL_NAME_ORDER:# + COIL_NAME_ORDER[4:]:
        sel_coil = COIL_NAME_ORDER.pop()
        print(sel_coil)
        sel_mat_file = sel_coil + "_ProcessedData.mat"
        # mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
        del visual_obj
        del mat_reader
        mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
        # SO here I need to enter the optimal file string....
        visual_obj = StoreOptimizeData(ddest=ddest, mat_reader=mat_reader,
                                       full_mask=full_mask, type_mask=type_mask)
        data_obj = OptimizeData(ddest=ddest, objective_str=objective_str,
                                mat_reader=mat_reader, full_mask=full_mask, type_mask=type_mask)
        metric_dict = data_obj.get_result_container(visual_obj.optimal_shim_rf_factor)
        del metric_dict['opt_shim']
        metric_dataframe = pd.DataFrame(metric_dict, index=[sel_coil])
        #
        if pd_dataframe is not None:
            pd_dataframe = pd.concat([pd_dataframe, metric_dataframe])
        else:
            pd_dataframe = metric_dataframe
        del data_obj
        print(metric_dataframe.round(4)[['power_deposition', 'peak_SAR', 'b1p_cov']])#.to_csv(sep='\t'))


print(pd_dataframe.T.round(4).to_csv(sep='\t'))
