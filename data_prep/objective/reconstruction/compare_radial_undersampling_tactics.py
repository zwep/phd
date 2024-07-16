import numpy as np
import sigpy
import skimage.data
import helper.plot_class as hplotc
import helper.array_transf as harray
from objective_helper.reconstruction import circus_radial_mask, CIRCUS_to_spokes
from objective_helper.reconstruction import undersample_img, undersample_trajectory, undersample_to_kspace
from objective_configuration.reconstruction import DPLOT


"""
We got this radial mask function from the /direct package

But how does it compare against interpolated radial data?

"""


# Load the data
A = skimage.data.astronaut()[:, :, 0]
A_scaled = harray.scale_minmax(A)
A_shape = A.shape
A_kspace = harray.transform_image_to_kspace_fftn(A)

"""
Apply CIRCUS mask
"""

circus_kspace_mask_5 = circus_radial_mask(A_shape, acceleration=5)
circus_kspace_mask_10 = circus_radial_mask(A_shape, acceleration=10)

A_CIRCUS_5 = harray.transform_kspace_to_image_fftn(A_kspace * circus_kspace_mask_5)
A_CIRCUS_10 = harray.transform_kspace_to_image_fftn(A_kspace * circus_kspace_mask_10)

"""
Apply radial trajectory NUFFT
"""

n_points = A_shape[0]
us_radial_traj_5 = undersample_trajectory(img_size=A_shape, n_points=n_points, p_undersample=100//5)
us_radial_traj_10 = undersample_trajectory(img_size=A_shape, n_points=n_points, p_undersample=100//10)

A_us_radial_5 = undersample_img(A_scaled[None], traj=us_radial_traj_5)[0]
A_us_radial_10 = undersample_img(A_scaled[None], traj=us_radial_traj_10)[0]

"""
Obtain the approximate spokes form the CIRCUS mask and see what the effect is of interpolation
"""

circus_spokes_5 = CIRCUS_to_spokes(circus_kspace_mask_5)
circus_spokes_10 = CIRCUS_to_spokes(circus_kspace_mask_10)

A_circus_spokes_5 = undersample_img(A_scaled[None], traj=circus_spokes_5, width=4, ovs=1.25)[0]
A_circus_spokes_10 = undersample_img(A_scaled[None], traj=circus_spokes_10, width=4, ovs=1.25)[0]

"""
Another option:

What if I discretize the gridded spokes so that I can multiply it..
"""


def binary_discretize_spokes(coordinates, shape):
    nx, ny = shape
    discretize_spokes = np.zeros((nx, ny))
    for i_coord in coordinates.astype(int):
        ix, iy = i_coord
        discretize_spokes[min(nx-1, ix + nx//2), min(ny-1, iy + ny//2)] = 1
    return discretize_spokes


us_radial_traj_discretized_5 = binary_discretize_spokes(us_radial_traj_5, shape=circus_kspace_mask_5.shape)
us_radial_traj_discretized_10 = binary_discretize_spokes(us_radial_traj_10, shape=circus_kspace_mask_10.shape)

A_us_radial_discretized_5 = harray.transform_kspace_to_image_fftn(A_kspace * us_radial_traj_discretized_5)
A_us_radial_discretized_10 = harray.transform_kspace_to_image_fftn(A_kspace * us_radial_traj_discretized_10)


"""
What if I grid the trajectory and mulitply that

This means a bit more sophisticated than the discretized version above..
"""


def gridded_spokes(coordinates, shape):
    nx, ny = shape
    n_points, ndim = coordinates.shape
    data = np.ones((1, n_points))
    res = sigpy.gridding(input=data, coord=coordinates, shape=shape)
    x_range = np.arange(-nx//2, nx //2)
    X, Y = np.meshgrid(x_range, x_range)
    dcf = np.sqrt(X ** 2 + Y ** 2)
    return np.fft.fftshift(res), dcf


gridded_traj_5, dcf_5 = gridded_spokes(us_radial_traj_5, circus_kspace_mask_5.shape)
gridded_traj_10, dcf_10 = gridded_spokes(us_radial_traj_10, circus_kspace_mask_10.shape)

A_grid_traj_5 = harray.transform_kspace_to_image_fftn(A_kspace * (gridded_traj_5 * (dcf_5)))
A_grid_traj_10 = harray.transform_kspace_to_image_fftn(A_kspace * gridded_traj_10 * (dcf_10))


"""
Now what if we project each spoke individually then sum..
This should be identical to nufft gridding
"""

import sigpy.mri
import os
img_size = A_shape
max_spokes = int(max(img_size) * np.pi / 2)
n_points = A_shape[0]
trajectory_radial = sigpy.mri.radial(coord_shape=(max_spokes, n_points, 2), img_shape=img_size)
undersampled_spokes = np.array([undersample_img((A_scaled + 1j * A_scaled)[None], traj=x)[0] for x in trajectory_radial])


# Woo summing also works in k-space
undersampled_spokes_fft = np.array([np.fft.fft2(x) for x in undersampled_spokes])
res = undersampled_spokes_fft.sum(axis=0)
res_img = np.fft.ifft2(res)
hplotc.ListPlot(res_img)
#
# spoke_selector = np.zeros(max_spokes)
# spoke_selector[1::5] = 1
# spoke_selector_matrix = np.diag(spoke_selector)
# res = np.einsum("xy, xnm->ynm", spoke_selector_matrix, undersampled_spokes)
A_individual_spoke_5 = undersampled_spokes[1::5].sum(axis=0)
A_individual_spoke_10 = undersampled_spokes[::10].sum(axis=0)
#

"""
Plot the result of using difference sampling schemes
"""
plot_array_5 = [A_CIRCUS_5, A_us_radial_5, A_circus_spokes_5, A_us_radial_discretized_5, A_grid_traj_5, A_individual_spoke_5]
plot_array_10 = [A_CIRCUS_10, A_us_radial_10, A_circus_spokes_10, A_us_radial_discretized_10, A_grid_traj_10, A_individual_spoke_10]

for ii, plot_array in enumerate([plot_array_5, plot_array_10]):
    plot_array = [harray.scale_minmax(np.abs(x)) for x in plot_array]
    patch_size = 64
    vmax_list = [(0, harray.get_proper_scaled_v2(x, (patch_size, patch_size), patch_size // 2)) for x in plot_array]
    subtitle_list = ['CIRCUS mask', 'Radial', 'CIRCUS spokes', 'Radial discretized binary', 'Gridded spokes', 'Seperated radial spokes']
    subtitle_list = [[x] for x in subtitle_list]
    plot_obj = hplotc.ListPlot(plot_array, ax_off=True, vmin=vmax_list, col_row=(3, 2), subtitle=subtitle_list, figsize=(30, 10))
    # plot_obj = hplotc.ListPlot(plot_array, ax_off=True, col_row=(3, 2), subtitle=subtitle_list,
    #                            figsize=(30, 10))
    plot_obj.figure.savefig(os.path.join(DPLOT, f'multiplicative_undersampling_strats_{5 * ii + 5}.png'))


"""
Below we visualize the input, mask and output for all examples...
"""

hplotc.ListPlot(A, ax_off=True)

hplotc.ListPlot([A, circus_kspace_mask_5, A_CIRCUS_5], ax_off=True, col_row=(3, 1))

plot_obj = hplotc.ListPlot([A, np.ones(A.shape), A_us_radial_5], ax_off=True, col_row=(3, 1))
plot_obj.ax_list[1].scatter(us_radial_traj_5[:, 0] + A.shape[0]//2, us_radial_traj_5[:, 1]+ A.shape[0]//2, alpha=0.5, c='w')

plot_obj = hplotc.ListPlot([A, np.ones(A.shape), A_circus_spokes_5], ax_off=True, col_row=(3, 1))
plot_obj.ax_list[1].scatter(circus_spokes_5[:, 0] + A.shape[0]//2, circus_spokes_5[:, 1]+ A.shape[0]//2, alpha=0.5, c='w')

hplotc.ListPlot([A, us_radial_traj_discretized_5, A_us_radial_discretized_5], ax_off=True, col_row=(3, 1))

hplotc.ListPlot([A, gridded_traj_5, A_grid_traj_5], ax_off=True, col_row=(3, 1))

hplotc.ListPlot([A, gridded_traj_5, A_grid_traj_5], ax_off=True, col_row=(3, 1))

