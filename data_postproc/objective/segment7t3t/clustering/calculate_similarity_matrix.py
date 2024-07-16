
"""
This was a trial for the clustering of the data...

It did not work
"""

resize_shape = (256, 256)
import os
import numpy as np
import helper.array_transf as harray

from skimage.metrics import structural_similarity
import scipy.stats
import helper.metric as hmetric
from data_postproc.objective.segment7t3t.clustering.FileGatherer import FileGather

file_gather_obj = FileGather()
# This uses the full images
# data_array = file_gather_obj.array_list
# prefix = ''
# ThIs uses the cropped out cardiac
data_array = file_gather_obj.cropped_array_list
ddata = '/data/seb/data/similarity_matrix'
prefix = 'cropped_'
dest_ssim = os.path.join(ddata, f'{prefix}ssim_matrix.npy')
dest_contrast = os.path.join(ddata, f'{prefix}contrast_matrix.npy')
dest_wss = os.path.join(ddata, f'{prefix}wss_matrix.npy')

with open('/data/seb/data/similarity_matrix/key_length_string.txt', 'w') as f:
    f.write(file_gather_obj.string_array_key)

n_img = data_array.shape[0]
ssim_matrix = np.zeros((n_img, n_img))
contrast_matrix = np.zeros((n_img, n_img))
wss_matrix = np.zeros((n_img, n_img))

i_img = 10
j_img = 230
for i_img in range(n_img):
    print(f"{i_img} / {n_img}", end='\r')
    x_i = data_array[i_img]
    x_i = harray.scale_minmax(x_i)
    x_i_mean = x_i.mean()
    x_hist_i, _ = np.histogram(x_i.ravel(), bins=256, range=(0, 1))
    for j_img in range(i_img, n_img):
        x_j = data_array[j_img]
        x_j = harray.scale_minmax(x_j)
        x_j_mean = x_j.mean()
        x_j = x_j * (x_i_mean / x_j_mean)
        x_hist_j, _ = np.histogram(x_j.ravel(), bins=256, range=(0, 1))
        wss_value = scipy.stats.wasserstein_distance(x_hist_i, x_hist_j)
        ssim_value = structural_similarity(x_i, x_j)
        contrast_value = hmetric.get_contrast_ssim(x_i, x_j)
        ssim_matrix[i_img, j_img] = ssim_value
        contrast_matrix[i_img, j_img] = contrast_value
        wss_matrix[i_img, j_img] = wss_value

# Misc
structural_similarity(np.rot90(x_i, k=2), x_j)
structural_similarity(np.rot90(x_i, k=3), x_j)
import helper.plot_class as hplotc
fig_obj = hplotc.ListPlot([np.rot90(x_i, k=3), x_j])
fig_obj.figure.savefig('/data/seb/est.png')

#
ddata = '/data/seb/data/similarity_matrix'
np.save(dest_ssim, ssim_matrix)
np.save(dest_contrast, contrast_matrix)
np.save(dest_wss, wss_matrix)

with open(os.path.join(ddata, 'key_length_string.txt'), 'w') as f:
    f.write(file_gather_obj.string_array_key)
