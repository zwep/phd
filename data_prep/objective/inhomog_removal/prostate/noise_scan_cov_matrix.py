import numpy as np
import os
import h5py
from skimage import img_as_int
import scipy.io
import helper.array_transf as harray
import matplotlib.pyplot as plt
import helper.plot_class as hplotc

"""
Read in the noise scan... get a covariance matrix..
"""

ddata = '/media/bugger/MyBook/data/7T_scan/cardiac/2022_02_28/ca_32383/matdata'
ddata_local = '/home/bugger/Documents/paper/inhomogeneity removal/noise_cov_matrix'
dest_cov = os.path.join(ddata, 'noise_cov.npy')
dest_cov_local = os.path.join(ddata_local, 'noise_cov.npy')

ddata_noise_scan = os.path.join(ddata, 'ca_28022022_1829142_5_1_transverse_radial_retrospectiveV4.lab.mat')
mat_obj = scipy.io.loadmat(ddata_noise_scan)
noise_vector = mat_obj['unsorted_data'][4][0]
n_points, n_coils = noise_vector.shape
cov_noise = (noise_vector.conjugate().T @ noise_vector) / n_coils
cov_noise = harray.scale_minmax(cov_noise, is_complex=True)
# Select only the last 8 coils
cov_noise = cov_noise[-8:, -8:]
np.save(dest_cov, cov_noise)
np.save(dest_cov_local, cov_noise)