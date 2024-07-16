import tooling.shimming.b1shimming_single as mb1
import numpy as np
import helper.plot_class as hplotc
import helper.plot_fun as hplotf
import helper.misc as hmisc
import helper.array_transf as harray
from objective_configuration.thesis import DPLOT
import os

"""

"""

dd = os.path.join(DPLOT, 'SAR/Head')
os.makedirs(dd, exist_ok=True)


dfile = '/home/bugger/Documents/paper/14T/plot_body_thomas_mask_rmse_power/8 Channel Dipole Array 7T/opt_sar_head_sar_40_degrees.npy'
SAR = hmisc.load_array(dfile)

MID_SLICE_OFFSET = (0, 0, -8)
SAR_slices = hplotf.get_all_mid_slices(SAR[::-1], offset=MID_SLICE_OFFSET)
hplotc.ListPlot(SAR_slices)

cropped_SAR = [np.ma.masked_array(x, mask=x == 0) for x in SAR_slices]


plot_obj = hplotc.ListPlot([x[None] for x in cropped_SAR], cmap='viridis', ax_off=True, col_row=(3, 1), proper_scaling=False, wspace=0, hspace=0,
                           figsize=(16, 8), proper_scaling_patch_shape=16, proper_scaling_stride=12, cbar=False)

plot_obj.savefig(os.path.join(dd, 'example_simulation_sar'), home=False)
