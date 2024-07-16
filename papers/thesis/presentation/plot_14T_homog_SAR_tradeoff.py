"""
Here we plot important images of all coils together..
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, COIL_NAME_ORDER, COLOR_MAP, DPLOT_FINAL, RF_SCALING_FACTOR, MID_SLICE_OFFSET, \
    COIL_NAME_ORDER_TRANSLATED
from objective_helper.fourteenT import ReadMatData, VisualizeData, StoreOptimizeData
import os
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import numpy as np

file_list = os.listdir(DDATA)
mat_files = [x for x in file_list if x.endswith('mat')]

str_normalization = '_head_sar'
str_appendix = ''

store_b1 = []
store_sar = []
sar_cor = []

for i_options in CALC_OPTIONS[0:1]:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    #
    # sel_coil_ind = COIL_NAME_ORDER.index('15 Channel Dipole Array')
    # sel_coil = COIL_NAME_ORDER[sel_coil_ind]
    fig, ax = plt.subplots()
    for sel_coil in COIL_NAME_ORDER[3:4]:
        sel_mat_file = sel_coil + "_ProcessedData.mat"
        mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
        visual_obj = StoreOptimizeData(ddest=ddest, mat_reader=mat_reader,
                                       full_mask=full_mask, type_mask=type_mask)

        ax.plot(visual_obj.result_dict['b1p_nrmse'], np.array(visual_obj.result_dict['peak_SAR']) * RF_SCALING_FACTOR ** 2, label=sel_coil)

        n_shim = 25
        for sel_index in [2, 15]:
            i_opt_shim = visual_obj.result_dict['opt_shim'][sel_index]

            b1p_shimmed, _, _ = visual_obj.get_shimmed_b1p(i_opt_shim)
            b1_plot_list = hplotf.get_all_mid_slices(b1p_shimmed[::-1], offset=MID_SLICE_OFFSET)
            transverse_slice_b1 = b1_plot_list[-1]
            #
            sar_shimmed, cor_sar, _ = visual_obj.get_shimmed_sar_q(i_opt_shim)
            sar_plot_list = hplotf.get_all_mid_slices(sar_shimmed[::-1], offset=MID_SLICE_OFFSET)
            transverse_slice_sar = sar_plot_list[-1]

            store_sar.append(transverse_slice_sar)
            store_b1.append(transverse_slice_b1)
            sar_cor.append(cor_sar)


import helper.misc as hmisc

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

# sel_index = 0
sel_index = 1
if sel_index == 0:
    title = 'High SAR, low error'
    vmax = 20
else:
    title = 'Low SAR, high error'
    vmax = 5

# title=''
plot_obj = hplotc.ListPlot([store_sar[sel_index].real * RF_SCALING_FACTOR ** 2, np.abs(store_b1[sel_index]  * RF_SCALING_FACTOR - 1 * RF_SCALING_FACTOR) / RF_SCALING_FACTOR], cbar=True, col_row=(2,1),
                vmin=[(0, vmax), (0, 1)], ax_off=True, cmap='viridis', title=title, dpi=100, wspace=0.2, hspace=0)
plot_obj.ax_list[0].set_title('SAR (W/kg) distribution ')
plot_obj.ax_list[1].set_title('Relative error with target B1')

