
"""
Sometimes we get images in the int8, int16, .., int64 range
If we normalize this with ITS OWN VALUES.. we can create a nice floating image out of it. But is that correct?

What if we FIRST normalize it with the range of the dtype, and THEN normalize. What are the effects?


There is hardly any difference....
"""

import numpy as np
import helper.array_transf as harray
import h5py
import helper.plot_class as hplotc
import matplotlib.pyplot as plt

ddata = '/home/bugger/Documents/data/check_clinic_registration/target_clean/M23_to_46_MR_20200925_0002.h5'

mean_diff_list = []
for i in range(100):
    with h5py.File(ddata, 'r') as f:
        A = np.array(f['data'][0])

    # normalization first
    A_minmax = harray.scale_minmax(A)

    # First dtype normalize
    max_dtype = np.iinfo(A.dtype).max
    A_dtype_scale = A / max_dtype
    A_dtype_minmax = harray.scale_minmax(A_dtype_scale)

    difference_array = A_minmax - A_dtype_minmax
    mean_diff = np.mean(difference_array)
    mean_diff_list.append(mean_diff)


# Visualize
plt.plot(mean_diff_list)

# Visualize the arrays
hplotc.ListPlot([A_minmax, A_dtype_minmax, difference_array], cbar=True)

# Visualize the distrutions
plt.hist(A_minmax.ravel(), color='r', alpha=0.5, bins=256)
plt.hist(A_dtype_minmax.ravel(), color='b', alpha=0.5, bins=256)