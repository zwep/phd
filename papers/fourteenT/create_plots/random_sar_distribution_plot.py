from objective_helper.fourteenT import VisualizeAllMetrics
from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, COIL_NAME_ORDER, DPLOT_FINAL, COIL_NAME_ORDER_TRANSLATOR
import os
import numpy as np


for i_options in CALC_OPTIONS:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    #
    # Here we visualize the results of one spoke
    visual_obj = VisualizeAllMetrics(ddest)
    fig_obj = visual_obj.visualize_hist_sar()
    fig_obj.tight_layout()
    fig_obj.savefig(os.path.join(DPLOT_FINAL, f'peak_sar_histogram2.png'), bbox_inches='tight')
    # fig_obj = visual_obj.visualize_hist_sar_single_figure()
    # fig_obj.savefig(os.path.join(DPLOT_FINAL, f'peak_sar_histogram_single.png'), bbox_inches='tight')
    # Print the number of exceedings of SAR
    if visual_obj.random_json_data is not None:
        max_points = 10000
        for ii, sel_coil in enumerate(COIL_NAME_ORDER):
            coil_plot_name = COIL_NAME_ORDER_TRANSLATOR[sel_coil]
            hist_data = visual_obj.random_json_data[sel_coil]['peak_SAR_normalized'][:max_points]
            print(sel_coil, np.mean(hist_data))
            perc_over = sum([x > 20 for x in hist_data]) / max_points
            print(sel_coil, perc_over)
