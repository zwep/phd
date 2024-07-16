import matplotlib.pyplot as plt

import os
import numpy as np
import torch
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.misc as hmisc
import objective.undersampled_recon.executor_undersampled_recon as executor
import helper.array_transf as harray
import reconstruction.ReadCpx as read_cpx

"""
Here we are going to reoncsturct an image by sampling the radial spokes OFF CENTER
Inspired by the measuremetns done with Edwin, we want to see how BAD an image can get

Load Cardiac image (one frame)
FFT
Sample with perfect radial spokes
Re-grid with OFF CENTER radial spokes
IFFT

"""


dir_radial = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_06/V9_16936'
high_time_radial = [os.path.join(dir_radial, x) for x in os.listdir(dir_radial) if 'high_time' in x and x.endswith('cpx')]
survey_files = [os.path.join(dir_radial, x) for x in os.listdir(dir_radial) if 'survey' in x and x.endswith('cpx')]

sel_file = survey_files[0]
cpx_obj = read_cpx.ReadCpx(sel_file)
cpx_array = cpx_obj.get_cpx_img()
cpx_array = np.rot90(cpx_array, axes=(-2, -1), k=1)
sel_loc = 1
n_coils = 8
cpx_array = np.squeeze(cpx_array[-n_coils:, sel_loc])
# Inspect loaded data
hplotc.ListPlot([cpx_array], augm='np.abs', start_square_level=2)

cpx_comb_coils = cpx_array.sum(axis=0)
cpx_kspace = harray.transform_image_to_kspace_fftn(cpx_comb_coils, dim=(-2, -1))

"""
Create perfect spokes
"""

from pynufft import NUFFT_cpu
import helper.nufft_recon as hnufft

# Define the re-sampling strategies
n_spokes = 128  # Number of spokes
Nd = (484, 428)  # image size
Kd = (512, 512)  # k-space size
Jd = (4, 4)  # interpolation size
y_line = np.zeros(Kd[1])
x_line = np.linspace(-np.pi, np.pi, Kd[0])
step_size = np.diff(x_line)[0]
single_line = np.vstack([x_line, y_line]).T
# Define the trajectory...
om_radial = hnufft.get_rot_stack(single_line, n_spokes)
om_undersampled = np.copy(om_radial)
om_undersampled = om_undersampled.reshape(n_spokes, Kd[0], 2)
om_undersampled[np.random.randint(n_spokes, size=n_spokes - n_spokes//6)] = 0
om_undersampled = om_undersampled.reshape(-1, 2)

# Transform the image space to k-space and check how that looks...
NufftObj = NUFFT_cpu()
NufftObj.plan(om_radial, Nd, Kd, Jd)
kspace0 = NufftObj.forward(cpx_comb_coils)
# I believe that we see what we want to see here...
hplotc.ListPlot(kspace0.reshape((n_spokes, -1)), augm='np.abs')

# Now create a perturbed version of om_radial..
# First reshape so that we can perturb each spoke
y_perturbed = np.deg2rad(1) * np.exp(-np.exp(1) * x_line ** 2) * np.sin(4*np.exp(1)*x_line)
perturbed_single_line = np.vstack([x_line, y_perturbed]).T
om_perturbed_radial = hnufft.get_rot_stack(perturbed_single_line, n_spokes)

nufft_normal = NUFFT_cpu()
nufft_normal.plan(om_radial, Nd, Kd, Jd)
image_normal = nufft_normal.solve(kspace0, solver='cg', maxiter=20)

nufft_undersampled = NUFFT_cpu()
nufft_undersampled.plan(om_undersampled, Nd, Kd, Jd)
image_undersampled = nufft_undersampled.solve(kspace0, solver='cg', maxiter=20)

nufft_perturbed = NUFFT_cpu()
nufft_perturbed.plan(om_perturbed_radial, Nd, Kd, Jd)
image_perturbed = nufft_perturbed.solve(kspace0, solver='cg', maxiter=20)

hplotc.ListPlot([[image_normal, image_perturbed, image_undersampled]], augm='np.abs', subtitle=[['normal', 'perturbed', 'undersampled']])
