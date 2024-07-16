import pandas as pd
import subprocess
import numpy as np
import os
import helper.plot_class as hplotc
import helper.misc as hmisc
import reconstruction.ReadCpx as read_cpx
import h5py

"""
Here we load the .cpx data and send it to a server
"""

ddata = '/home/bugger/Documents/paper/b1_shim/b1_shim_series.csv'
ddest = '/media/bugger/MyBook/data/7T_data/b1_shim_series'
ddest_remote = 'seb@legolas.bmt.tue.nl:/data/seb/data/b1_shim'
#  This removes the appendices that we added...
processed_files = ['_'.join(hmisc.get_base_name(x).split('_')[:-2]) for x in os.listdir(ddest)]

pd_dataframe = pd.read_csv(ddata)
for i, i_row in pd_dataframe.iterrows():
    i_dir = i_row['directory']
    i_file = i_row['file_name']
    if i_file not in processed_files:
        print('Starting ', i_file)
        file_path = os.path.join(i_dir, i_file)
        cpx_obj = read_cpx.ReadCpx(file_path)
        cpx_shim_array = cpx_obj.get_cpx_img()
        n_coil_x, n_loc, n_dyn, _, _, n_coil_y, _, nx, ny = cpx_shim_array.shape
        if n_coil_x == 8 and n_coil_y == 8:
            for i_loc in range(n_loc):
                for i_dyn in range(n_dyn):
                    dest_file_name = i_file + f"_{i_loc}" + f"_{i_dyn}.h5"
                    dest_file_path = os.path.join(ddest, dest_file_name)
                    sel_cpx_array = cpx_shim_array[:, i_loc, i_dyn, 0, 0, :, 0]
                    sel_cpx_array_real = sel_cpx_array.real.astype(np.float32)
                    sel_cpx_array_imag = sel_cpx_array.imag.astype(np.float32)
                    sel_cpx_array_stack = np.stack([sel_cpx_array_real, sel_cpx_array_imag])
                    print(sel_cpx_array_stack.shape)
                    with h5py.File(dest_file_path, 'w') as f:
                        f.create_dataset('data', data=sel_cpx_array_stack)

        del cpx_obj
        del cpx_shim_array
    else:
        print('Done ', i_file)


    # for i_file in os.listdir(ddest):
    #     source_file = os.path.join(ddest, i_file)
    #     dest_file = os.path.join(ddest_remote)
    #     cmd_line = f'scp {source_file} {dest_file}'
    #     print(cmd_line)
    #     subprocess.check_output(cmd_line)