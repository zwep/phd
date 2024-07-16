"""
Here we plot important images of all coils together..
"""
from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, COIL_NAME_ORDER, COLOR_MAP, DPLOT_FINAL, MID_SLICE_OFFSET, \
    COIL_NAME_ORDER_TRANSLATED
from objective_helper.fourteenT import ReadMatData, VisualizeData
import os
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import numpy as np

file_list = os.listdir(DDATA)
mat_files = [x for x in file_list if x.endswith('mat')]

str_normalization = '_head_sar'
str_appendix = ''

for i_options in CALC_OPTIONS[0:1]:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
#
    plot_array_sar = []
    sel_mat_file = mat_files[0]
    for sel_coil in COIL_NAME_ORDER:
        sel_mat_file = sel_coil + "_ProcessedData.mat"
        mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
        visual_obj = VisualizeData(ddest=ddest, mat_reader=mat_reader, full_mask=full_mask, type_mask=type_mask)
        #
        coil_name = visual_obj.mat_reader.coil_name
        #
        sar_array = np.load(visual_obj.path_sar_file + str_normalization + str_appendix + '.npy')
        sar_plot_list = hplotf.get_all_mid_slices(sar_array[::-1], offset=MID_SLICE_OFFSET)
        plot_array_sar.append(sar_plot_list[-1])
#
    # Flip the head with ::-1 to have a different orientation
    plot_array_sar = np.array(plot_array_sar)[:, ::-1]
    vmax = np.max(plot_array_sar.real)
    fig_obj = hplotc.ListPlot(plot_array_sar[None], subtitle=[COIL_NAME_ORDER_TRANSLATED],
                              augm='np.abs', ax_off=True, sub_col_row=(3, 2), wspace=0,
                              cmap=COLOR_MAP, vmin=(0, int(vmax)))
    fig_obj.figure.savefig(os.path.join(DPLOT_FINAL, f'sar_rf_shim.png'), bbox_inches='tight', pad_inches=0)
    fig_obj = hplotc.ListPlot(plot_array_sar[None], subtitle=[COIL_NAME_ORDER_TRANSLATED],
                              cbar=True, augm='np.abs', ax_off=True, sub_col_row=(3, 2), wspace=0,
                              cmap=COLOR_MAP, vmin=(0, int(vmax)))
    fig_obj.figure.savefig(os.path.join(DPLOT_FINAL, f'cbar_sar_rf_shim.png'), bbox_inches='tight')


