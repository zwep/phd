"""
Wanted to make a seperate file for this...
this inolved the moving of SA data

either unfolded by me.. or by reconframe


"""
import nrrd
import itertools
import scipy.io
import re
import helper.array_transf as harray
import helper.plot_class as hplotc

import os
import numpy as np


ddata_npy = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac'
# Deze directory bestaat niet meer als het goed is. Dit zou de nieuwe moeten zijn
# /media/bugger/MyBook/data/7T_data/unfolded_cardiac
ddata_mat = '/media/bugger/MyBook/data/7T_data/cardiac/unfolded_per_v_number'

dest_dir = '/media/bugger/MyBook/data/7T_data/unfolded_sa'

# Get all the directories with files....
file_dict = {}
total_sum = 0
for d, _, f in os.walk(ddata_npy):
    scan_type = os.path.basename(d)
    v_number = os.path.basename(os.path.dirname(d))
    filter_list = [os.path.join(d, x) for x in f if x.endswith('.npy')]
    n_files = len(filter_list)
    print(f'Found {scan_type} scan: {v_number}')
    print('\t Number of files ', len(filter_list))
    if len(filter_list) > 0:
        total_sum += len(filter_list)
        file_dict.setdefault(v_number, {})
        file_dict[v_number].setdefault(scan_type, [])
        file_dict[v_number][scan_type] = filter_list

# Get all the .mat files
res = []
for d, _, f in os.walk(ddata_mat):
    f_filter = [x for x in f if (x.endswith('mat') and ('saV4' in x))]
    if len(f_filter):
        f_filter = [os.path.join(d, x) for x in f_filter]
        res.append(f_filter)

# load the .mat files into an array
mat_sa_files = list(itertools.chain(*res))
for sel_file in mat_sa_files:
    file_name_no_ext = os.path.splitext(os.path.basename(sel_file))[0]
    print(file_name_no_ext)
    dest_file_path = os.path.join(dest_dir, file_name_no_ext + '.nrrd')
    mat_obj = scipy.io.loadmat(sel_file)
    data_array = mat_obj.get('reconstructed_data', None)
    data_array = np.moveaxis(np.moveaxis(np.squeeze(data_array), -2, 0), -1, 0)
    data_array_stack = harray.to_stacked(data_array)
    nrrd.writer.write(dest_file_path, data_array_stack.astype(np.float32))

# Load the data from the v numbers we are still missing
v_numbers_sa_mat = [re.findall('(V9_[0-9]+)', x)[0] for x in mat_sa_files]
unique_v_numbers_sa_mat = list(set(v_numbers_sa_mat))
v_numbers_not_present_in_mat = [x for x in list(file_dict.keys()) if x not in unique_v_numbers_sa_mat]

for i_vnumber in v_numbers_not_present_in_mat:
    sa_files = file_dict[i_vnumber].get('sa', [])
    if len(sa_files):
        for sel_file in sa_files:
            file_name_no_ext = os.path.splitext(os.path.basename(sel_file))[0]
            print(file_name_no_ext)
            dest_file_path = os.path.join(dest_dir, file_name_no_ext + '.nrrd')
            A = np.load(sel_file)
            A = np.squeeze(A)
            if A.ndim == 4:
                pass
            elif A.ndim == 3:
                A = A[None]
            elif A.ndim == 2:
                A = A[None, None]
            else:
                print('errr')

            data_array_stack = harray.to_stacked(A)
            nrrd.writer.write(dest_file_path, data_array_stack.astype(np.float32))


