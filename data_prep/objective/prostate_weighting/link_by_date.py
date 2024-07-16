
import numpy as np
import os
import h5py
import helper.plot_class as hplotc
import matplotlib.pyplot as plt

"""
I want to see if we can match by date....
"""

dMR = '/local_scratch/sharreve/mri_data/prostate_h5'
dest_dir = '/local_scratch/sharreve'
for i_patient in sorted(os.listdir(dMR)):
    print(i_patient)
    patient_dir = os.path.join(dMR, i_patient)
    patient_list = os.listdir(patient_dir)
    if 'MRL' in patient_list and 'MRI' in patient_list:
        MRL_dir = os.path.join(patient_dir, 'MRL')
        MRI_dir = os.path.join(patient_dir, 'MRI')
        MRL_files = [x for x in os.listdir(MRL_dir) if x.endswith('h5')]
        MRI_files = [x for x in os.listdir(MRI_dir) if x.endswith('h5')]
        MRL_dates = [x[:8] for x in MRL_files]
        MRI_dates = [x[:8] for x in MRI_files]
        intersection_MRL_MRI = set(MRL_dates).intersection(set(MRI_dates))
        MRL_files_sel = [os.path.join(MRL_dir, x) for x in MRL_files if x[:8] in intersection_MRL_MRI]
        MRI_files_sel = [os.path.join(MRI_dir, x) for x in MRI_files if x[:8] in intersection_MRL_MRI]
        for i_file in MRL_files_sel + MRI_files_sel:
            with h5py.File(i_file, 'r') as f:
                n_slice = f['data'].shape[0]
                A = np.array(f['data'][n_slice//2])
            # print(i_file, n_slice)
            fig_obj = hplotc.ListPlot(A)
            base_name = os.path.basename(i_file)
            base_name, _ = os.path.splitext(base_name)
            dir_name = os.path.basename(os.path.dirname(i_file))
            fig_obj.figure.savefig(f'/local_scratch/sharreve/{i_patient}_{dir_name}_{base_name}.png')
            hplotc.close_all()
