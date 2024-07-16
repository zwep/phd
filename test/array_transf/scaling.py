import helper.array_transf as harray
import numpy as np
import torchio
import skimage.data

"""
Here we visualize the intermediate steps of the get proper scaling v2 algorithm

"""


dummy_array = np.ones((256, 256))
astronaut = skimage.data.astronaut()[:, :, 0]
n_fields = 5


poly_5_biasfield_list = []
for _ in range(n_fields):
    gen_biasf_obj = torchio.transforms.RandomBiasField(coefficients=0.8, order=6)
    gen_biasf = gen_biasf_obj(astronaut[None, :, :, None])[0, :, :, 0]
    gen_biasf = harray.scale_minmax(gen_biasf)
    poly_5_biasfield_list.append(gen_biasf)

import helper.plot_class as hplotc


poly_5_biasfield_list = harray.scale_minmax(poly_5_biasfield_list)
vmax = [(0, harray.get_proper_scaled_v2(x, patch_shape=128, stride=16)) for x in poly_5_biasfield_list]

mean_scaling = [(0, x) for x in np.mean(poly_5_biasfield_list, axis=(1,2))]
hplotc.ListPlot([poly_5_biasfield_list], ax_off=True, cbar=True)

hplotc.ListPlot([harray.scale_mean_std(x) for x in poly_5_biasfield_list], ax_off=True, cbar=True)
hplotc.ListPlot([poly_5_biasfield_list], ax_off=True, cbar=True, vmin=[mean_scaling])
hplotc.ListPlot([poly_5_biasfield_list], vmin=[vmax], ax_off=True, cbar=True, cbar_round_n=6, wspace=0.2)


x = poly_5_biasfield_list[0]
patch_shape = 128
stride = 16

hplotc.ListPlot([x, x], cbar=True, col_row=(2, 1))
max_value = 1
# Make sure that the images are scaled
# We are always dealing with 2D images
x = harray.scale_minmax(x)
temp_patches = harray.get_patches(x, patch_shape=patch_shape, stride=stride)
# Subset those with enough non-zero values in their patches
temp_patches = np.array([x for x in temp_patches if (np.isclose(x, 0, atol=1e-5)).sum() / np.prod(x.shape) < 0.5])

hplotc.ListPlot(temp_patches[:12], col_row=(3,4), ax_off=True, cbar=True)

max_patches = temp_patches.max(axis=(-2, -1))
import matplotlib.pyplot as plt

max_patches_mean = max_patches.mean()
plt.plot(max_patches)
plt.hlines(max_patches_mean, 0, len(max_patches), 'r')
# Why do I take the mean of x here..?
# I think this should be identical..
# max_mean_patches_index = [(i, x.mean()) for i, x in enumerate(max_patches) if x.mean() > max_patches.mean()]
# Was it also possible to take the mean not over the maximum of each patch.. but the patches self?

# Here we select patches that have a larger mean than the average maximum value
max_mean_patches_index = [(i, x) for i, x in enumerate(max_patches) if x > max_patches_mean]

# hplotc.ListPlot([x[1] for x in max_mean_patches_index[:12]], col_row=(3, 4))

if len(max_mean_patches_index):
    # Sort based on the maximum value
    sorted_max_mean_patches_index = sorted(max_mean_patches_index, key=lambda x: x[1])
    # Then take the lower 20% of this distribution
    n_max_sel = int(len(sorted_max_mean_patches_index) * 0.2)
    patch_index, _ = zip(*sorted_max_mean_patches_index[:n_max_sel])
    plt.plot([x[1] for x in sorted_max_mean_patches_index])
    plt.vlines(n_max_sel, 0.9*min(max_mean_patches_index)[1], 1, color='k')
    patch_index = list(patch_index)
    sel_patches = temp_patches[patch_index]
    # Finally.. calculate the mean of this selections + 2 standard deviations (pretty arbitrary)
    max_value = sel_patches.mean() + 2 * sel_patches.std()

