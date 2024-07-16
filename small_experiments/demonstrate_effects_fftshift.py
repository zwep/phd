import numpy as np
import matplotlib.pyplot as plt
import helper.plot_class as hplotc
from scipy.fft import fft2, ifft2
import skimage.data
from scipy.fft import fftshift, ifftshift, fft2, ifft2

import reconstruction.ReadCpx as read_cpx
import h5py
import helper.array_transf as harray
import helper.misc as hmisc

"""
This shows that there is very little effect of doing a fftshift

The most important part is of course to un-do it afterwards
But that is what I normally already did


"""
# Create a simple test image
# Image 1
ddata = '/media/bugger/MyBook/data/7T_data/radial_retro_cardiac_cine/mat_data/transverse/v9_05032022_1032049_8_1_transverse_radial_retrospectiveV4.mat'
A = hmisc.load_array(ddata)
image = np.squeeze(A.T)[0].sum(axis=0)
# Image 2
# image = skimage.data.astronaut()[:, :, 0]
# Image 3
# image = np.zeros((256, 256))
# image[100:150, 100:150] = 255

# Perform Fourier transform without Fourier shift
spectrum_fft = fft2(image)

# Perform inverse Fourier transform
restored_image_fft = ifft2(spectrum_fft)

# Display the original and restored images
hplotc.ListPlot([image, restored_image_fft], augm='np.abs')
hplotc.ListPlot([image, restored_image_fft], augm='np.angle')

# Perform Fourier shift
shifted_image = fftshift(image)

# Perform Fourier transform
spectrum = fft2(shifted_image)

# Perform inverse Fourier transform
restored_shifted_image = ifft2(spectrum)

# Undo Fourier shift
restored_image = ifftshift(restored_shifted_image)


hplotc.ListPlot([image, shifted_image, restored_shifted_image, restored_image], augm='np.abs')
hplotc.ListPlot([image, shifted_image, restored_shifted_image, restored_image], augm='np.angle')


hplotc.ListPlot([spectrum_fft, spectrum, spectrum_fft - spectrum], augm='np.abs')
hplotc.ListPlot([spectrum_fft, spectrum, spectrum_fft - spectrum], augm='np.angle')

fig, ax = plt.subplots()
ax.hist(np.angle(spectrum_fft).ravel(), label='sepctrum fft', alpha=0.2)
ax.hist(np.angle(spectrum).ravel(), label='sepctrum', alpha=0.2)

fig, ax = plt.subplots(2)
ax[0].hist(np.abs(spectrum_fft).ravel(), label='sepctrum fft', alpha=0.5, bins=256, range=(0, 0.1))
ax[1].hist(np.abs(spectrum).ravel(), label='sepctrum', alpha=0.5, bins=256, range=(0, 0.1))


"""
Now lets explore some re-gridding of kspace..?
"""

import numpy as np
kspace = fft2(image)
kspace_shift = fftshift(kspace)
kspace_padded = np.pad(kspace_shift, ((50, 50), (50, 50)))
kspace_ishift = ifftshift(kspace_padded)
image_padded = ifft2(kspace_ishift)
hplotc.ListPlot([image, kspace, kspace_shift, kspace_padded, kspace_ishift, image_padded], augm='np.abs')
hplotc.ListPlot([image, image_padded], augm='np.abs')