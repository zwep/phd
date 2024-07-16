import numpy as np
import h5py
import helper.plot_class as hplotc
import os

"""
Check if we made any weird errors...
"""

d_input = '/home/bugger/Documents/data/check_clinic_registration/input/M18_to_16_MR_20210108_0002.h5'
d_target = '/home/bugger/Documents/data/check_clinic_registration/target/M18_to_16_MR_20210108_0002.h5'
d_mask = '/home/bugger/Documents/data/check_clinic_registration/mask/M18_to_16_MR_20210108_0002.h5'
d_clean = '/home/bugger/Documents/data/check_clinic_registration/target_clean/M18_to_16_MR_20210108_0002.h5'

# Load mask
with h5py.File(d_mask, 'r') as h5_obj:
    mask_array = np.array(h5_obj['data'])

sel_d = d_input
with h5py.File(sel_d, 'r') as h5_obj:
    input_array = np.array(h5_obj['data'])



# hplotc.SlidingPlot(input_array)
# hplotc.SlidingPlot(input_array.sum(axis=1))

import helper.array_transf as harray
temp_A = np.abs(input_array.sum(axis=1))[0]
temp_B = mask_array[0]
temp_mask = harray.get_treshold_label_mask(temp_A)
import helper.misc as hmisc
dice_mean = hmisc.dice_metric(temp_mask, temp_B)
hplotc.ListPlot([temp_mask, temp_B])

# Now check everything together
res = []
for sel_d in [d_input, d_target, d_mask, d_clean]:
    with h5py.File(sel_d, 'r') as h5_obj:
        input_array = np.array(h5_obj['data'][40])

    if input_array.ndim == 2:
        res.append(input_array)
    else:
        res.append(np.abs(input_array.sum(axis=0)))

hplotc.ListPlot(res)
hplotc.ListPlot(np.prod(np.array(res), axis=0))
hplotc.ListPlot(np.prod(np.array(res)[np.array([0, 1, -1])], axis=0))
