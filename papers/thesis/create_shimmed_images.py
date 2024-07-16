import tooling.shimming.b1shimming_single as mb1
import numpy as np
import helper.plot_class as hplotc
import helper.misc as hmisc
import helper.array_transf as harray
from objective_configuration.thesis import DPLOT
import os

"""

"""

dd = os.path.join(DPLOT, 'Shimming/Prostate')
os.makedirs(dd, exist_ok=True)

ddata = '/media/bugger/MyBook/data/simulated/prostate/transmit_flavio'
sel_file = hmisc.get_single_file(ddata, 'npy')
A = hmisc.load_array(sel_file)
A = A * 1e6

body_mask = np.abs(A).sum(axis=0) > 0
body_mask = harray.resize_and_crop(body_mask, 1.4)
A_cropped = np.array([harray.get_crop(np.abs(x), body_mask)[0] for x in A])

A_masked = np.ma.masked_array(A_cropped, mask=A_cropped == 0)

plot_obj = hplotc.ListPlot(A_masked, cmap='viridis', ax_off=True, col_row=(4, 2), proper_scaling=True, wspace=0, hspace=0,
                           figsize=(16, 2))

plot_obj.savefig(os.path.join(dd, 'example_simulation_b1'), home=False, bbox_inches=None)


# This is to make sure that the MaskCreator tool gets the right input
pre_mask = A.sum(axis=0)
# mask_handle = hplotc.MaskCreator(pre_mask)
# sel_mask = mask_handle.mask
sel_mask = harray.create_random_center_mask(pre_mask.shape, mask_fraction=0.05, random=False, y_offset=-10)

# Initiate the shimming procedure
shim_proc = mb1.ShimmingProcedure(A, sel_mask, str_objective='b1', relative_phase=True)
opt_shim, opt_value = shim_proc.find_optimum(opt_method='Nelder-Mead', verbose=True, maxiter=5000)

shimmed_A = harray.apply_shim(A, cpx_shim=opt_shim)
abs_shimmed = np.abs(shimmed_A)
mean_value = np.mean(abs_shimmed[sel_mask==1])
scale_factor = np.pi/2 * (1/mean_value)
signal = np.sin(scale_factor * abs_shimmed) ** 3
hplotc.ListPlot([abs_shimmed, signal])


cropped_shim, _ = harray.get_crop(abs_shimmed, body_mask)
cropped_no_shim, _ = harray.get_crop(A.sum(axis=0), body_mask)
cropped_mask, _ = harray.get_crop(sel_mask, body_mask)
#

cropped_no_shim_masked = np.ma.masked_array(cropped_no_shim, mask=cropped_no_shim == 0)
cropped_shim_masked = np.ma.masked_array(cropped_shim, mask=cropped_shim == 0)

plot_obj = hplotc.ListPlot([cropped_no_shim_masked[None], cropped_shim_masked[None] * (cropped_mask + 1) ], augm='np.abs', ax_off=True, col_row=(2, 1),
                proper_scaling=True, proper_scaling_patch_shape=64, cmap='viridis', figsize=(8,4), hspace=0.01)

plot_obj.savefig(os.path.join(dd, 'example_simulation_b1_shimmed'), home=False)
