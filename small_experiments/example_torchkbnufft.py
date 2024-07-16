import torch
import sigpy.mri
import helper.array_transf as harray
import torchkbnufft

import helper.plot_class as hplotc
import torchkbnufft as tkbn
import numpy as np
from skimage.data import shepp_logan_phantom

x = shepp_logan_phantom().astype(complex)
x_range = np.linspace(0, 4 * np.pi, 400)
X, Y = np.meshgrid(x_range, x_range)
Z = np.sin(X) + np.cos(Y)
hplotc.ListPlot(Z)

x = x * np.exp(1j * Z)
im_size = x.shape
# convert to tensor, unsqueeze batch and coil dimension
# output size: (1, 1, ny, nx)
x = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(torch.complex64)

max_spokes, n_points, ndim = (120, 480, 2)
trajectory_radial = sigpy.mri.radial(coord_shape=(max_spokes, n_points, 2), img_shape=im_size)

trajectory_radial = 2 * np.pi * harray.scale_minmax(trajectory_radial) - np.pi
ktraj = torch.tensor(np.moveaxis(trajectory_radial.reshape(-1, 2), -1, 0)).to(torch.float)
dcf = tkbn.calc_density_compensation_function(ktraj, im_size=im_size)
# test = dcf.reshape(max_spokes, n_points)
# hplotc.ListPlot(test.numpy())
nufft_ob = tkbn.KbNufft(im_size=im_size)
adnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size)
# outputs a (1, 1, klength) vector of k-space data
kdata = nufft_ob.forward(x, ktraj)
rec_x = adnufft_ob.forward(kdata * dcf, ktraj)
hplotc.ListPlot([x, rec_x.numpy()])
hplotc.ListPlot([x, rec_x.numpy()], augm='np.angle')
hplotc.ListPlot(x, augm='np.angle')
hplotc.ListPlot(x, augm='np.abs')

"""
"""

import torch
import torchkbnufft

# Generate complex phantom data
phantom_size = (64, 64)  # Size of the phantom image
num_samples = 1000  # Number of non-uniform samples
k_space_size = phantom_size + (2,)  # Size of the k-space

phantom = torch.randn(phantom_size, dtype=torch.complex64)

# Generate radial trajectory
angles = torch.linspace(0, 2 * torch.pi, num_samples)  # Angles for radial spokes
radius = torch.linspace(0, 0.5, num_samples)  # Radius values for radial trajectory
k_space_traj = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)

# Create the NUFFT object
nufft = torchkbnufft.KbNufft(im_size=phantom_size)

# Forward NUFFT: Phantom to k-space
k_space = nufft.forward(phantom.unsqueeze(0), k_space_traj.T)

# Backward NUFFT: k-space to phantom
reconstructed_phantom = nufft.backward(k_space, k_space_traj.unsqueeze(0))

# Verify the reconstruction error
reconstruction_error = torch.norm(phantom.unsqueeze(0) - reconstructed_phantom) / torch.norm(phantom.unsqueeze(0))
print("Reconstruction error:", reconstruction_error.item())

