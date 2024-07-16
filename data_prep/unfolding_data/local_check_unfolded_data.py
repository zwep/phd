"""
We ve create a bunch of unfolded data... lets check em out!
"""

import helper.array_transf as harray
import helper.plot_class as hplotc

import os
import numpy as np

ddata = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac'
# Get all the directories with files....
file_dict = {}
for d, _, f in os.walk(ddata):
    scan_type = os.path.basename(d)
    v_number = os.path.basename(os.path.dirname(d))
    filter_list = [os.path.join(d, x) for x in f if x.endswith('.npy')]
    n_files = len(filter_list)
    print(f'Found {scan_type} scan: {v_number}')
    print('\t Number of files ', len(filter_list))
    if len(filter_list) > 0:
        file_dict.setdefault(v_number, {})
        file_dict[v_number].setdefault(scan_type, [])
        file_dict[v_number][scan_type] = filter_list

print(f'\n\nNumber of Vnumbers ', len(file_dict))


hplotc.close_all()
# Process each patient, per MRI/MRL, per file
scan_type = 'sa'
v_number = 'V9_13975'
temp_dict = file_dict[v_number]
for v_number, temp_dict in file_dict.items():
    print('Processing ', scan_type, v_number)
    for sel_file in file_dict[v_number].get(scan_type, []):
        A = np.load(sel_file)
        n_take = A.ndim
        for i in range(0, n_take-2):
            A = np.take(A, 0, axis=0)

        print('\t\t Shape of array ', A.shape)

        hplotc.ListPlot(A, augm='np.abs', title=sel_file)
