# encoding: utf-8

"""
Transforming the h5 files to have an extra undersampling dimension.
"""

import numpy as np
import os
from pynufft import NUFFT_cpu
import helper.nufft_recon as hnufft
import h5py
import time

# Define the re-sampling strategies
NufftObj = NUFFT_cpu()
Nd = (256, 256)  # image size
Kd = (512, 512)  # k-space size
Jd = (6, 6)  # interpolation size
y_line = np.zeros(Kd[0])
x_line = np.linspace(-np.pi, np.pi, Kd[0])
single_line = np.vstack([x_line, y_line]).T


# Amount of samples from the undersampled trajectories
n_random_uc = 4
# Amount of lines used
n_undersampled = 15
N_lines = 8 * n_undersampled
om_ga_star = hnufft.get_golden_angle_rot_stack(single_line, N_lines)
print('original shape', om_ga_star.shape)


# Define source directory... (target data is fully sampled complex valued)
# data_dir = '/home/bugger/Documents/data/semireal/prostate_simulation_h5/train/target'
data_dir = '/data/seb/semireal/prostate_simulation_h5/test/target'
# data_dir = '/data/seb/semireal/prostate_simulation_h5/train/target'

# Target directory
dest_dir = f'/data/seb/semireal/prostate_simulation_h5/test/input_{n_undersampled}'
# dest_dir = f'/data/seb/semireal/prostate_simulation_h5/train/input_{n_undersampled}'

if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

file_list = [x for x in os.listdir(data_dir) if x.endswith('.h5')]

for i_file in file_list:
    orig_file = os.path.join(data_dir, i_file)
    dest_file = os.path.join(dest_dir, i_file)

    # Load the fully sampled .h5 file
    with h5py.File(orig_file, 'r') as f:
        orig_data = np.array(f['data'])

    data_shape = orig_data.shape
    n_phase = data_shape[0]
    n_slice = data_shape[1]

    # Dit is denk ik ook niet slim....
    new_shape = (4,) + data_shape
    uc_data = np.empty(new_shape, dtype=np.complex)

    # Perform the undersampling for all views...
    t0 = time.time()
    for i_phase in range(n_phase):
        for i_slice in range(5):
            for i_uc in range(n_random_uc):
                temp_data = orig_data[i_phase, i_slice]
                om_undersampled = hnufft.get_undersampled_traj(om_ga_star, n_undersampled=n_undersampled, total_lines=N_lines)
                temp_data_sampled, res_ksp = hnufft.nufft_to_image(temp_data, om_undersampled, Nd=Nd, Kd=Kd, Jd=Jd)
                uc_data[i_uc, i_phase, i_slice] = temp_data_sampled
        print(f'phase: {i_phase} time: {time.time() - t0}', end='\n')

    with h5py.File(dest_file, 'w') as f:
        f.create_dataset('data', data=uc_data)
