
"""
Transform some of the target images to radial k-space in a more undersampled way..

This was used to load all the numpy files.. undersample then.. and then continue.
But I found that that creates LOADS of numpy files. It should've been more organised
So I switched to .h5.
"""

import numpy as np
import os
import SimpleITK as sitk
import nrrd
import re
import sys
from pynufft import NUFFT_cpu
import helper.nufft_recon as hnufft

# Resampling strategies
NufftObj = NUFFT_cpu()

Nd = (256, 256)  # image size
Kd = (512, 512)  # k-space size
Jd = (6, 6)  # interpolation size

y_line = np.zeros(Kd[0])
x_line = np.linspace(-np.pi, np.pi, Kd[0])
single_line = np.vstack([x_line, y_line]).T
n_undersampled = 15
N_lines = 8 * n_undersampled
om_ga_star = hnufft.get_golden_angle_rot_stack(single_line, N_lines)
print('original shape', om_ga_star.shape)

# dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation'
dir_data = '/data/seb/semireal/prostate_simulation'

for d, _, f in os.walk(dir_data):
    d_name = os.path.basename(d)
    if d_name == 'target':
        new_dir = f'input_{n_undersampled}'
        dest_dir = os.path.join(os.path.dirname(d), new_dir)
        print('\t Base directory', dest_dir, d_name)

        if not os.path.isdir(dest_dir):
            print('Creating direcotry ', dest_dir)
            os.mkdir(dest_dir)

        file_list = [x for x in os.listdir(d) if x.endswith('.h5')]
        for i_file in file_list[0:1]:
            file_dir = os.path.join(d, i_file)
            file_name, ext = os.path.splitext(i_file)

            target_array = np.load(file_dir)

            # Get several variations on the undersampling...
            # Since we get these randomly
            # Increase data with a factor of 4..
            for i_uc in range(4):
                file_name_target = file_name + f'_uc_{i_uc}' + ext
                dest_file_input = os.path.join(dest_dir, file_name_target)
                om_undersampled = hnufft.get_undersampled_traj(om_ga_star, n_undersampled=n_undersampled, total_lines=N_lines)
                target_array_sampled, res_ksp = hnufft.nufft_to_image(target_array, om_undersampled, Nd=Nd, Kd=Kd, Jd=Jd)

                np.save(dest_file_input, target_array_sampled)
