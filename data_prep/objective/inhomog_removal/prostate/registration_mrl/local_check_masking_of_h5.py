"""

Some masks were not filled ocrrectly..

check the process here
"""

import helper.plot_class as hplotc
import h5py
import numpy as np
import os
import helper.array_transf as harray


# One example directory of a fault mask.
# Others are recorded in a Excel file
ddata = '/home/bugger/Documents/data/1.5T/prostate_mri_mrl/38_MR/MRL'
list_files = os.listdir(ddata)

list_files = sorted(list_files)
sel_file = list_files[0]
for sel_file in list_files:
    file_dir = os.path.join(ddata, sel_file)
    h5_obj = h5py.File(file_dir, 'r')
    A = np.array(h5_obj['data'])
    print('Shape of array ', A.shape, sel_file)

    n_slice, _, _ = A.shape
    first_array = A[0]
    middle_array = A[n_slice // 2]
    last_array = A[-1]
    A_mask = harray.get_treshold_label_mask(middle_array, treshold_value=np.mean(A) * 0.5)

    # A_mask = harray.get_treshold_label_mask(middle_array, treshold_value=np.mean(A) * 0.8)

    hplotc.ListPlot([A[n_slice//2], A_mask])