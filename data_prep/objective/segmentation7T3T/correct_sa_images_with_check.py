"""
Now that we've corrected the images in a way..
We are going to correct them
"""
import helper.array_transf as harray
import numpy as np
import helper.plot_class as hplotc
import scipy.ndimage
import os
import helper.plot_class
from pandas_ods_reader import read_ods
import nrrd

dest_dir = '/media/bugger/MyBook/data/7T_data/unfolded_sa_filtered'
d_csv = os.path.join(dest_dir, 'selection_overview.ods')

hplotc.close_all()
A_csv = read_ods(d_csv, 'Sheet1')
for i, i_row in A_csv.iterrows():
    print('Processing ', i_row['file_name'])
    exclude_all = i_row['exclude_all']
    if exclude_all:
        print('\t Not using this file')
        continue
    else:
        file_dir = i_row['file_name']
        file_name = os.path.basename(file_dir)
        loaded_array, _ = nrrd.read(file_dir)
        array_shape = loaded_array.shape
        n_loc = array_shape[0]
        ignore_slice = i_row['exclude_loc']

        if ignore_slice is not None:
            if isinstance(ignore_slice, float):
                ignore_slice = [int(ignore_slice)]
            else:
                ignore_slice = [int(x) for x in ignore_slice.split(',')]

            sel_slice = np.array([x for x in range(n_loc) if x not in ignore_slice])
            loaded_array = loaded_array[sel_slice]

        loaded_array = harray.to_complex(loaded_array)
        print('Shape of object: ', loaded_array.shape)
        n_loc = loaded_array.shape[0]
        n_card = loaded_array.shape[1]
        rot = i_row['rot']
        if not np.isnan(rot):
            print('Rotating with ', rot)
            loaded_array = harray.rotate_complex(loaded_array, axes=(-2, -1), angle=int(rot), reshape=True)

        # hplotc.ListPlot(loaded_array[:, 0], augm='np.abs')
        temp_A = harray.to_stacked(loaded_array)
        nrrd.writer.write(os.path.join(dest_dir, file_name), temp_A.astype(np.float32))