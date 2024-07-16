import helper.phantom as hphantom
import numpy as np
import sigpy.mri
import matplotlib.pyplot as plt
import helper.plot_class as hplotc

A = hphantom.phantom()

fully_sampled_radial = np.copy(A)

img_shape = fully_sampled_radial.shape[-2:]

n_points = max(img_shape)
n_spokes = int(np.pi / 2 * max(img_shape))
width = 6
ovs = 1.25

# Define undersampled trajectory, the same for ALL the coils
# Create a new trajectory...
trajectory_radial = sigpy.mri.radial(coord_shape=(n_spokes, n_points, 2), img_shape=img_shape, golden=False)
for i_angle in range(100):
    plt.scatter(trajectory_radial[i_angle][:, 0], trajectory_radial[i_angle][:, 1])
trajectory_radial = trajectory_radial.reshape(-1, 2)

trajectory_radial.min()

temp_kspace = sigpy.nufft(fully_sampled_radial, coord=trajectory_radial, width=width, oversamp=ovs)
temp_kspace = temp_kspace.reshape(n_points, n_spokes)

plt.plot(np.abs(temp_kspace[0:2*n_points]))
hplotc.ListPlot(np.tile(np.fft.fft(temp_kspace[0]), 256).T.reshape(256, 256), augm='np.abs')
