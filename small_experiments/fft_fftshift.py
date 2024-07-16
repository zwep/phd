import skimage.data
import numpy as np
from objective_helper.reconstruction import step_by_step_plot

# Get one coil image...?

A = skimage.data.astronaut()[:, :, 0]
step_by_step_plot(np.fft.fftshift(np.fft.fft2(A)))

A_rec = np.fft.ifft2((np.fft.fftshift(np.fft.fft2(A))))
A_rec1 = np.fft.ifft2((np.fft.ifftshift(np.fft.fft2(A))))
import helper.plot_class as hplotc
hplotc.ListPlot([A - A_rec ,A - A_rec1, A])


dd = '/media/bugger/MyBook/data/7T_data/cardiac_radial_p2ch/v9_02052021_0933560_11_3_p2ch_radialV4.npy'
A = np.load(dd)
sel_coil = A[0][0]
step_by_step_plot(np.fft.fft2(sel_coil))