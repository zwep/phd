"""
Quickly compare the images sizes of 1.5T and 3T stuff..?

"""

import numpy as np
import os
import h5py
import helper.plot_class as hplotc

dMRI = '/local_scratch/sharreve/mri_data/prostate_h5/1_MR/MRI'
dMRL = '/local_scratch/sharreve/mri_data/prostate_h5/1_MR/MRL'

print("MRI")
MRI_examples = []
for i_file in os.listdir(dMRI):
    if i_file.endswith('h5'):
        dpath = os.path.join(dMRI, i_file)
        print(dpath)
        with h5py.File(dpath, 'r') as f:
            n_slice = f['data'].shape[0]
            print(f['data'].shape)
            mid_slice = np.array(f['data'][n_slice//2])
        MRI_examples.append(mid_slice)

fig_obj = hplotc.ListPlot(MRI_examples)
fig_obj.figure.savefig(os.path.join(dMRI, 'example_images.jpg'))


MRL_examples = []
print("MRL")
for i_file in os.listdir(dMRL):
    if i_file.endswith('h5'):
        dpath = os.path.join(dMRL, i_file)
        print(dpath)
        with h5py.File(dpath, 'r') as f:
            print(f['data'].shape)
            n_slice = f['data'].shape[0]
            mid_slice = np.array(f['data'][n_slice // 2])
        MRL_examples.append(mid_slice)

fig_obj = hplotc.ListPlot(MRL_examples)
fig_obj.figure.savefig(os.path.join(dMRL, 'example_images.jpg'))



"""
MORE
"""

import matplotlib.pyplot as plt
#dMR = '/local_scratch/sharreve/mri_data/mask_h5'
dMR = '/local_scratch/sharreve/mri_data/prostate_h5'
filter_field_strength = 'MRL'

MRI_max_examples = []
for d, subdir, f in os.walk(dMR):
    if filter_field_strength in d:
        filter_f = [x for x in f if x.endswith('h5')]
        if len(filter_f):
            print('subdir', subdir)
            MRI_examples = []
            for i_file in filter_f:
                dpath = os.path.join(d, i_file)
                # print(dpath)
                with h5py.File(dpath, 'r') as f:
                    n_slice = f['data'].shape[0]
                    mid_slice = np.array(f['data'][n_slice//2])
                MRI_examples.append(mid_slice)
                MRI_max_examples.append(n_slice)
            print('Plotting...')
            fig_obj = hplotc.ListPlot([[x.astype(int) for x in MRI_examples]], start_square_level=2, figsize=(20, 20))
            fig_obj.figure.savefig(os.path.join(d, 'example_images_bin.jpg'))
            hplotc.close_all()

