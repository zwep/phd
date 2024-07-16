import helper.misc as hmisc
import re
import helper.plot_class as hplotc
import itertools
from objective_helper.reconstruction import scan2direct_array, resize_array, mat2direct
import numpy as np
import os
import h5py


"""
Test
"""

ddata = '/media/bugger/MyBook/data/7T_data/'
dscan = '/media/bugger/MyBook/data/7T_scan/cardiac'
ddest = '/media/bugger/MyBook/data/7T_data/direct_inference_free'
if not os.path.isdir(ddest):
    os.makedirs(ddest)
re_v_number = re.compile('(V[0-9]_[0-9]*)')

retro_mat = hmisc.find_all_files_in_dir('free_breathing', dscan, ext='raw')
print(retro_mat)

# Now check which v-numbres belong to these
vnumber_dict = {}
for i_mat in retro_mat:
    # Filter on this, since we have duplicates otherwise..
    # if '/cardiac_cine_mat/' in i_mat:
    if '/cardiac_cine_mat/' in i_mat:
        date_time = hmisc.get_base_name(i_mat)[:19]
        found_files = hmisc.find_all_files_in_dir(date_time, dscan, ext='sin')
        if len(found_files):
            v_numbers = list(itertools.chain(*[re_v_number.findall(x) for x in found_files]))
            for i_vnum in v_numbers:
                vnumber_dict.setdefault(i_vnum, [])
                vnumber_dict[i_vnum].append(i_mat)

wrong_files = []
for vnum, vnum_files in vnumber_dict.items():
    for i_file in vnum_files:
        base_name = hmisc.get_base_name(i_file)
        anatomy = os.path.basename(os.path.dirname(i_file))
        A = hmisc.load_array(i_file, data_key='vartosave')
        print(A[0][0].shape)
        if isinstance(A, dict):
            wrong_files.append((vnum, i_file))
            continue

        new_file_name = f"{base_name[:19]}_{vnum}_{anatomy}.h5"
        ndim_array = np.squeeze(A).ndim
        if ndim_array == 4:
            A_input = mat2direct(A)
            del A
            A_input = resize_array(A_input)
            A_input = A_input[:, ::-1, ::-1]
            print(f"Storing {new_file_name}")
            with h5py.File(os.path.join(ddest, 'input', new_file_name), 'w') as f:
                f.create_dataset('kspace', data=A_input)
            del A_input
        else:
            wrong_files.append((vnum, i_file))