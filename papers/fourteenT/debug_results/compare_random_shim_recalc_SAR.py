"""
So we have a difference in SAR... between VOPs and the Q10g matrix

What is that about....
"""

from objective_helper.fourteenT import VisualizeAllMetrics
from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, COIL_NAME_ORDER, DPLOT_FINAL, COIL_NAME_ORDER_TRANSLATOR, RF_SCALING_FACTOR
import os
import numpy as np


for i_options in CALC_OPTIONS[0:1]:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    #
    # Here we visualize the results of one spoke
    visual_obj = VisualizeAllMetrics(ddest)
    visual_obj.random_json_data = visual_obj.load_json_files(subdir='random_shim_10_000_recalc_sar', cpx_key='random_shim')
    zz = visual_obj.random_json_data['8 Channel Loop Array big']['peak_SAR']
    print(np.min(zz), np.mean(zz), np.max(zz))
    fig_obj = visual_obj.visualize_hist_sar()
    fig_obj.savefig(os.path.join(DPLOT_FINAL, f'peak_sar_histogram_compared1.png'), bbox_inches='tight')
    visual_obj.random_json_data = visual_obj.load_json_files(subdir='random_shim_10_000', cpx_key='random_shim')
    zz = np.array(visual_obj.random_json_data['8 Channel Loop Array big']['peak_SAR']) * RF_SCALING_FACTOR ** 2
    print(np.min(zz), np.mean(zz), np.max(zz))
    fig_obj = visual_obj.visualize_hist_sar()
    fig_obj.savefig(os.path.join(DPLOT_FINAL, f'peak_sar_histogram_compared2.png'), bbox_inches='tight')
