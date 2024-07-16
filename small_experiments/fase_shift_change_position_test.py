import helper.array_transf as harray
import skimage.data
import helper.plot_class as hplotc
import numpy as np

A = skimage.data.astronaut()[:, :, 0]

X, Y = np.meshgrid(np.arange(-256, 256), np.arange(-256, 256))
A_kspc = harray.transform_image_to_kspace_fftn(A)
shift_z = (np.sqrt(X ** 2 + Y ** 2))
shift_z = 2 * harray.scale_minmax(shift_z) - 1
A_kspc_shift = A_kspc * np.exp(2 * np.pi * 1j * shift_z)
A_shift = harray.transform_kspace_to_image_fftn(A_kspc_shift)

hplotc.ListPlot([A_kspc, A_kspc_shift, A_kspc - A_kspc_shift], augm='np.angle')
# Funny.. we made an edge detector.
hplotc.ListPlot([A, A_shift, A - A_shift], augm='np.abs')


# Now try to get a lower resolution thing...?
A_kspc = harray.transform_image_to_kspace_fftn(A)
kspace_shape = A_kspc.shape
center_x, center_y = np.array(kspace_shape) // 2
delta_x = center_x // 6
delta_y = center_y // 6

A_kspc[:(center_x - delta_x), :] = 0
A_kspc[(center_x + delta_x):, :] = 0
A_kspc[:, (center_y + delta_y):] = 0
A_kspc[:, :(center_y - delta_y)] = 0

center_kspace = np.ones((2 * delta_x, 2 * delta_y), dtype=np.complex)
center_kspace = A_kspc[(center_x - delta_x):(center_x + delta_x), (center_y - delta_y):(center_y + delta_y)]

A_shift = harray.transform_kspace_to_image_fftn(A_kspc)
B_shift = harray.transform_kspace_to_image_fftn(center_kspace)
hplotc.ListPlot([A_kspc==0,B_shift,  A_shift, A], augm='np.abs')