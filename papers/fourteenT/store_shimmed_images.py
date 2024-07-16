import matplotlib.pyplot as plt

from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, RF_SCALING_FACTOR, MID_SLICE_OFFSET, COLOR_MAP
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
from objective_helper.fourteenT import ReadMatData, StoreOptimizeData
import os

"""
Here we store the results of the RF shimming minimization

This requires that we have 
"""


file_list = os.listdir(DDATA)
mat_files = [x for x in file_list if x.endswith('mat')]

for i_options in CALC_OPTIONS[0:1]:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    # Doing so will leave us with a single instance of VisualizeData / ReadMatData
    visual_obj = None
    mat_reader = None
    for sel_mat_file in mat_files:
        # sel_mat_file = mat_files.pop()
        del visual_obj
        del mat_reader
        mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
        # SO here I need to enter the optimal file string....
        visual_obj = StoreOptimizeData(ddest=ddest, mat_reader=mat_reader, full_mask=full_mask, type_mask=type_mask)
        # visual_obj.store_SNR()
        # visual_obj.store_conductivity()
        # This shows the current B1 we get when shimming with our optimal scaling factor...
        # for optimal_shim in visual_obj.result_dict['opt_shim']:
        # # SH sept 2023: adding manual plotting of the B1 map
        # sel_shim = visual_obj.optimal_shim_rf_factor
        # b1p_shimmed, correction_head_sar, optm_power_deposition = visual_obj.get_shimmed_b1p(sel_shim)
        # # Checking on the B1+ CoV
        # import helper.metric as hmetric
        # sel_b1p = visual_obj.select_array(b1p_shimmed, mask_array=visual_obj.selected_mask)
        # fig, ax = plt.subplots()
        # ax.hist(sel_b1p)
        # ax.set_title(mat_reader.coil_name)
        # cov_b1p = hmetric.coefficient_of_variation(sel_b1p) * 100
        # print(mat_reader.coil_name, cov_b1p)
        # b1_plot_list = hplotf.get_all_mid_slices(b1p_shimmed[::-1], offset=MID_SLICE_OFFSET)
        # fig_obj = hplotc.ListPlot([b1_plot_list], ax_off=True, cbar=True, wspace=0.5, vmin=(0, 3.4),
        #                   title=visual_obj.mat_reader.coil_name, cmap=COLOR_MAP, figsize=(30, 10))
        # # Plotting the mask..
        # mask_vis = hplotf.get_all_mid_slices(visual_obj.selected_mask[::-1], offset=MID_SLICE_OFFSET)
        # _ = hplotc.ListPlot([mask_vis], ax_off=True, cbar=True, wspace=0.5, vmin=(0, 3.4),
        #                           title='mask', cmap=COLOR_MAP, figsize=(30, 10))
        # fig_obj.figure.savefig(visual_obj.path_b1_file + "_" + mat_reader.coil_name + '.png', bbox_inches='tight')
        # fig_obj.figure.savefig(visual_obj.path_b1_file + "_" + mat_reader.coil_name + '.svg', bbox_inches='tight')
        # b1p_shimmed = (visual_obj.b1_container['b1p'].T @ (RF_SCALING_FACTOR * optimal_shim)).T
        # print(np.abs(b1p_shimmed)[visual_obj.selected_mask==1].mean())
        # hplotc.SlidingPlot(b1p_shimmed)
        #
        #
        #b1p_shimmed, correction_head_sar, optm_power_deposition = visual_obj.get_shimmed_b1p(visual_obj.optimal_shim_rf_factor)
        #hplotc.ListPlot(hplotf.get_all_mid_slices(b1p_shimmed, offset=MID_SLICE_OFFSET))

        print(visual_obj.path_sar_file)
        visual_obj.store_optimal_b1()
        visual_obj.store_optimal_sar()



