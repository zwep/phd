import os
import numpy as np
from objective_helper.fourteenT import ReadMatData
from objective_configuration.fourteenT import DDATA, MID_SLICE_OFFSET, DMASK_THOMAS, DPLOT_FINAL
import helper.plot_fun as hplotf
import helper.plot_class as hplotc

# Load the mask..
thomas_mask = np.load(DMASK_THOMAS)

# Plot the mask
plot_array = hplotf.get_all_mid_slices(thomas_mask[::-1], offset=MID_SLICE_OFFSET)
fig_obj = hplotc.ListPlot([plot_array], ax_off=True)
fig_obj.figure.savefig(os.path.join(DPLOT_FINAL, f'mask_array.png'), bbox_inches='tight')

# Load the sigma/brain masks to be laid on top of it
mat_reader = ReadMatData(ddata=DDATA, mat_file='8 Channel Dipole Array_ProcessedData.mat')
mask_container = mat_reader.read_mask_object()
brain_mask = mask_container['target_mask'] - mask_container['substrate_mask']
brain_mid = hplotf.get_all_mid_slices(brain_mask[::-1], offset=MID_SLICE_OFFSET)
sigma_mid = hplotf.get_all_mid_slices(mask_container['sigma_mask'][::-1], offset=MID_SLICE_OFFSET)

# Add the different plots
new_plot_array = []
for ii in range(3):
    temp_img = 3 * plot_array[ii] + 2 * brain_mid[ii] - 1 * sigma_mid[ii]
    new_plot_array.append(temp_img)

# Flip the head orientation using ::1
new_plot_array[-1] = new_plot_array[-1][::-1]
fig_obj = hplotc.ListPlot([new_plot_array], ax_off=True)
for ii in range(2):
    img_shape = new_plot_array[ii].shape
    sel_slice = img_shape[0] // 2 + MID_SLICE_OFFSET[0]
    # Add a red line to denote the slice we take
    fig_obj.ax_list[ii].hlines(xmin=0, xmax=img_shape[1], y=sel_slice, colors='r')
# Store the image
fig_obj.figure.savefig(os.path.join(DPLOT_FINAL, f'mask_array_gray_scale.png'), bbox_inches='tight')

# Experimental: try RGB plotting..
dummy_zeros = np.zeros(plot_array[2].shape)
plot_array_rgb = np.stack([plot_array[2], dummy_zeros, dummy_zeros], axis=-1)[None]
brain_rgb = np.stack([dummy_zeros, brain_mid[2], dummy_zeros], axis=-1)[None]
sigma_rgb = np.stack([dummy_zeros, dummy_zeros, sigma_mid[2]], axis=-1)[None]
fig_obj = hplotc.ListPlot([plot_array_rgb + brain_rgb + sigma_rgb], cmap='rgb', sub_col_row=(1, 1))
fig_obj.figure.savefig(os.path.join(DPLOT_FINAL, f'mask_array_rgb_scale.png'), bbox_inches='tight')