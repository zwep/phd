import numpy as np
import sigpy
import objective_helper.reconstruction as objective_helper
import helper.plot_class as hplotc

"""
Simply take 
"""

cart_sampled_img, cart_sampled_kspace = objective_helper.get_cartesian_sampled_cardiac_data()
n_points, _ = cart_sampled_kspace.shape
img_shape = cart_sampled_kspace.shape
n_spokes = n_points
p_undersample = 25
width = 6
ovs = 2.5

# Define a trajectory
trajectory_radial = sigpy.mri.radial(coord_shape=(n_spokes, n_points, 2), img_shape=img_shape)
trajectory_radial = trajectory_radial.reshape(-1, 2)
dcf = np.sqrt(trajectory_radial[:, 0] ** 2 + trajectory_radial[:, 1] ** 2)

n_undersample = int((p_undersample / 100) * n_spokes)
undersampled_trajectory = np.array(np.split(trajectory_radial, n_spokes))
random_lines = np.random.choice(range(n_spokes), size=(n_spokes - n_undersample), replace=False)
undersampled_trajectory[random_lines] = None
undersampled_trajectory = undersampled_trajectory.reshape(-1, 2)

temp_kspace = sigpy.nufft(cart_sampled_img, coord=undersampled_trajectory, width=width, oversamp=ovs)
temp_img = sigpy.nufft_adjoint(temp_kspace * dcf, coord=undersampled_trajectory, oshape=img_shape, width=width, oversamp=ovs)

hplotc.ListPlot([cart_sampled_img, temp_img], augm='np.abs')