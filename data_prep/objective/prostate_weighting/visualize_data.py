
import h5py
import re
import scipy.io
import helper.array_transf as harray
import numpy as np
import helper.misc as hmisc
import os
import helper.plot_class as hplotc
import shutil

"""
First... visualize how severe the problem is..
 so we can see how well we can improve upon it
 
 Conclussion: For most of the images.. finding a linear function between the slice indices was enough
 Note that the used data directory (tconvert_h5) is outdated and no longer used
 Please use prostate_weighting_h5
"""

data_set_type = 'validation'
dd_input = f'/local_scratch/sharreve/mri_data/prostate_weighting_h5/{data_set_type}/input'
dd_target = f'/local_scratch/sharreve/mri_data/prostate_weighting_h5/{data_set_type}/target'
dd_target_cor = f'/local_scratch/sharreve/mri_data/prostate_weighting_h5/{data_set_type}/target_corrected'
dd_target_corn4 = f'/local_scratch/sharreve/mri_data/prostate_weighting_h5/{data_set_type}/target_corrected_N4'
dd_dest = f'/local_scratch/sharreve/mri_data/prostate_weighting_h5/visualize_{data_set_type}_target_proc'

if not os.path.isdir(dd_dest):
    os.makedirs(dd_dest)


file_names = os.listdir(dd_input)
sel_file = sorted(file_names)[0]
hplotc.close_all()

for sel_file in file_names:
    file_name, _ = os.path.splitext(sel_file)
    print('sel file', sel_file)
    input_path = os.path.join(dd_input, sel_file)
    target_path = os.path.join(dd_target, sel_file)
    target_cor_path = os.path.join(dd_target_cor, sel_file)
    target_corn4_path = os.path.join(dd_target_corn4, sel_file)

    with h5py.File(input_path, 'r') as f:
        n_slice_input = f['data'].shape[0]
        A0 = np.array(f['data'][n_slice_input // 2])
        A1 = np.array(f['data'][0])
        A2 = np.array(f['data'][-1])
    with h5py.File(target_path, 'r') as f:
        n_slice_target = f['data'].shape[0]
        B0 = np.array(f['data'][n_slice_target // 2])
        B1 = np.array(f['data'][0])
        B2 = np.array(f['data'][-1])
    if os.path.isfile(target_cor_path):
        with h5py.File(target_cor_path, 'r') as f:
            n_slice_target = f['data'].shape[0]
            C0 = np.array(f['data'][n_slice_target // 2])
            C1 = np.array(f['data'][0])
            C2 = np.array(f['data'][-1])
        plot_obj_C = hplotc.ListPlot([C0, C1, C2], cbar=True, vmin=(0,1))
        file_name_C = os.path.join(dd_dest, file_name + '_target_proc.png')
        plot_obj_C.figure.savefig(file_name_C)

    if os.path.isfile(target_corn4_path):
        with h5py.File(target_corn4_path, 'r') as f:
            n_slice_target = f['data'].shape[0]
            D0 = np.array(f['data'][n_slice_target // 2])
            D1 = np.array(f['data'][0])
            D2 = np.array(f['data'][-1])
        plot_obj_D = hplotc.ListPlot([D0, D1, D2], cbar=True)
        file_name_D = os.path.join(dd_dest, file_name + '_target_procn4.png')
        plot_obj_D.figure.savefig(file_name_D)

    plot_obj_A = hplotc.ListPlot([A0, A1, A2], cbar=True)
    plot_obj_B = hplotc.ListPlot([B0, B1, B2], cbar=True)
    file_name_A = os.path.join(dd_dest, file_name + '_input.png')
    file_name_B = os.path.join(dd_dest, file_name + '_target.png')
    plot_obj_A.figure.savefig(file_name_A)
    plot_obj_B.figure.savefig(file_name_B)
    hplotc.close_all()
