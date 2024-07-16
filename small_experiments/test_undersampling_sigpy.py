import helper.array_transf as harray
import skimage.transform as sktransf
import helper.phantom as hphantom
import helper.plot_class as hplotc
import sigpy
import matplotlib.pyplot as plt
import os
import numpy as np
import reconstruction.ReadCpx as read_cpx
import scipy.io
import helper.nufft_recon as hnufft


try:
    dd = '/media/bugger/MyBook/data/7T_data/cardiac_radial_transverse/v9_02052021_1431531_6_3_transradialfastV4.npy'
    A_img = np.load(dd)
    A_sel_img = A_img[:, 0]
    A_real = sktransf.resize(A_sel_img[-1].real, (256, 256), order=3)
    A_imag = sktransf.resize(A_sel_img[-1].imag, (256, 256), order=3)
    A_cpx = A_real + 1j * A_imag
except FileNotFoundError:
    A_sel_img = hphantom.phantom()
    A_cpx = A_sel_img + 1j * A_sel_img



n_points = 500
n_spokes = 250
n_undersample = 250
x_line = np.linspace(-256, 256, n_points)
y_line = np.zeros(n_points)
single_line = np.vstack([x_line, y_line]).T
coord = hnufft.get_golden_angle_rot_stack(single_line, n_spokes)
i_n, i_width, i_ovs = (n_points, 6, 1.5)
dcf = np.sqrt(coord[:, 0] ** 2 + coord[:, 1] ** 2)
A_kspace = sigpy.nufft(A_cpx, coord*dcf[:, None], width=i_width, oversamp=i_ovs)
res2 = sigpy.nufft_adjoint(A_kspace, coord*dcf[:, None], oshape=(344, 344), width=i_width, oversamp=i_ovs)
hplotc.ListPlot([A_cpx, res2], augm='np.abs')

import sigpy.mri
n_spokes = 256
n_points = 256
image_shape = (256, 256)
n_undersampled = 128

# Get the FULLY SAMPLED TRAJECTORY
# This will be used to resample the image space data...
om_radial = sigpy.mri.radial(coord_shape=(n_spokes, n_points, 2), img_shape=image_shape)
om_radial_reshape = om_radial.reshape(-1, 2)
# With associated DCF
dcf = np.sqrt(om_radial_reshape[:, 0] ** 2 + om_radial_reshape[:, 1] ** 2)

# Get the trajectory kspace..
A_kspace = sigpy.nufft(A_cpx, coord=om_radial_reshape)

# Create undersampled trajectory...
# Copy this...
x_traj_split = np.array(np.split(om_radial_reshape, n_spokes))
kspace_spokes = np.array(np.split(A_kspace, n_spokes))
hplotc.ListPlot(kspace_spokes[0])
random_lines = np.random.choice(range(n_spokes), size=(n_spokes - n_undersample), replace=False)
x_traj_split[random_lines] = None
x_traj_undersampled = x_traj_split.reshape(-1, 2)

# Undersampled trajectory
trajectory_img_space = sigpy.nufft_adjoint(A_kspace * dcf, coord=x_traj_undersampled, oshape=A_cpx.shape)

hplotc.ListPlot([trajectory_img_space, A_cpx], augm='np.abs')
