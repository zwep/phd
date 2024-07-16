from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, COIL_NAME_ORDER, COLOR_MAP
from objective_helper.fourteenT import ReadMatData, VisualizeData
import os
import helper.plot_class as hplotc
import numpy as np
import matplotlib.pyplot as plt

"""
Here we plot important images of all coils together..
"""


file_list = os.listdir(DDATA)
mat_files = [x for x in file_list if x.endswith('mat')]

str_normalization = '_head_sar'

for i_options in CALC_OPTIONS:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    axial_slice_dict = {}
    # Doing so will leave us with a single instance of VisualizeData / ReadMatData
    visual_obj = None
    mat_reader = None
    sel_mat_file = mat_files[0]
    for sel_mat_file in mat_files:
        del visual_obj
        del mat_reader
        mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
        visual_obj = VisualizeData(ddest=ddest, mat_reader=mat_reader, full_mask=full_mask, type_mask=type_mask)
        coil_name = visual_obj.mat_reader.coil_name
        _ = axial_slice_dict.setdefault(coil_name, {})
        visual_obj.plot_optimal_lambda()
        # Checking the metrics...
        fig, ax = plt.subplots()
        ax.scatter(visual_obj.result_dict['residual'], visual_obj.result_dict['peak_SAR'])
        # Leaving out the plotting of COND an SNR
        optimal_sar = visual_obj.plot_optimal_sar(str_normalization=str_normalization)
        optimal_b1 = visual_obj.plot_optimal_b1(str_normalization=str_normalization)
        #
        axial_slice_dict[coil_name]['b1p_axial'] = optimal_b1[-1]
        axial_slice_dict[coil_name]['sar_axial'] = optimal_sar[-1]
        hplotc.close_all()

    plot_array_b1 = [axial_slice_dict[sel_coil]['b1p_axial'] for sel_coil in COIL_NAME_ORDER]
    vmax = np.max(np.abs(np.array(plot_array_b1)))
    fig_obj = hplotc.ListPlot(np.array(plot_array_b1)[None], subtitle=[list(axial_slice_dict.keys())],
                              augm='np.abs', ax_off=True, sub_col_row=(3, 2), cmap=COLOR_MAP, wspace=0.2,
                              cbar=True, vmin=(0, vmax))
    if not os.path.isdir(visual_obj.path_b1_file):
        os.makedirs(visual_obj.path_b1_file)
    fig_obj.figure.savefig(os.path.join(visual_obj.path_b1_file, f'shimmed_b1p_{str_normalization}.png'), bbox_inches='tight')

    plot_array_sar = [axial_slice_dict[sel_coil]['sar_axial'] for sel_coil in COIL_NAME_ORDER]
    vmax = np.max(np.abs(np.array(plot_array_sar)))
    fig_obj = hplotc.ListPlot(np.array(plot_array_sar)[None], subtitle=[list(axial_slice_dict.keys())],
                              augm='np.abs', ax_off=True, sub_col_row=(3, 2), cbar=True, wspace=0.2,
                              cmap=COLOR_MAP, vmin=(0, vmax))
    if not os.path.isdir(visual_obj.path_sar_file):
        os.makedirs(visual_obj.path_sar_file)
    fig_obj.figure.savefig(os.path.join(visual_obj.path_sar_file, f'shimmed_sar_{str_normalization}.png'), bbox_inches='tight')
    hplotc.close_all()

