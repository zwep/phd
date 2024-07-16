import os
import helper.array_transf as harray
import numpy as np
import helper.misc as hmisc
import helper.plot_class as hplotc
import objective_helper.reconstruction as hrecon
from objective_configuration.reconstruction import DRESULT, DDATA, DINFERENCE
import sigpy.mri
import matplotlib.pyplot as plt

"""

We have some results...

For 0p we see very good results. Too good
For 0p on synth. u.s. data we see worse results. Really bad. Too bad..?

Why is this

- Checking conversion. Is the input properly formatted?
- Check scaling of the images. Are both synth u.s. and u.s. similar?
- Check config files. Which mask is used?


--> The reason was that .. fftshift screwed up the order of the coils.

"""
# # # Example training data

ddata_train = os.path.join(DDATA, 'mixed/train/input')
file_list = os.listdir(ddata_train)
sel_file = file_list[0]
file_path = os.path.join(ddata_train, sel_file)
A = hmisc.load_array(file_path, data_key='kspace', sel_slice='mid')

# Going to check the distribution
A_cpx = A[..., ::2] + 1j * A[..., 1::2]
A_cpx = np.moveaxis(A_cpx, -1, 0)
A_img = np.fft.ifft2(np.fft.fftshift(A_cpx, axes=(-2, -1)), norm='ortho')
print('Shape of cpx', A_cpx.shape)

A_sos = hrecon.convert_to_sos(A)
fig_obj = hplotc.ListPlot([A_sos], cbar=True)
fig_obj.figure.savefig(os.path.expanduser('~/test_input_train_sos.png'))

N = 256
max_spokes = int(np.ceil((np.pi / 2) * N))
n_points = N
img_size = (N, N)
trajectory_radial = sigpy.mri.radial(coord_shape=(max_spokes, n_points, 2), img_shape=img_size, golden=False)
us_trajectory = trajectory_radial[0::5]
us_img = hrecon.undersample_img(A_img, us_trajectory)

fig, ax = plt.subplots(3, 4)
fig2, ax2 = plt.subplots(3, 4)
ax = ax.ravel()
ax2 = ax2.ravel()
for i, i_ax in enumerate(ax):
    z_abs = np.abs(us_img)[-i]
    z_angle = np.angle(us_img)[-i]
    i_ax.imshow(z_abs)
    ax2[i].imshow(z_angle)

fig.savefig('/home/sharreve/synth_input_us_abs.png')
fig2.savefig('/home/sharreve/synth_input_us_angle.png')


# # # Example inf target

ddata_inf = os.path.join(DINFERENCE, 'target')
file_list = os.listdir(ddata_inf)
sel_file = file_list[0]
file_path = os.path.join(ddata_inf, sel_file)
A = hmisc.load_array(file_path, data_key='kspace', sel_slice='mid')


A_sos = hrecon.convert_to_sos(A)
fig_obj = hplotc.ListPlot([A_sos], cbar=True)
fig_obj.figure.savefig(os.path.expanduser('~/test_input_inf_sos.png'))


# Going to check the distribution
A_cpx = A[..., ::2] + 1j * A[..., 1::2]
A_cpx = np.moveaxis(A_cpx, -1, 0)
fig_obj = hplotc.ListPlot(A_cpx)
fig_obj.figure.savefig(os.path.expanduser('~/test_input_inf_kspace.png'))

A_img = np.fft.ifft2(A_cpx, norm='ortho')
# A_img = np.fft.ifft2(np.fft.fftshift(A_cpx, axes=(-2, -1)), norm='ortho')
print('Shape of cpx', A_cpx.shape)

N = 256
max_spokes = int(np.ceil((np.pi / 2) * N))
n_points = N
img_size = (N, N)
trajectory_radial = sigpy.mri.radial(coord_shape=(max_spokes, n_points, 2), img_shape=img_size, golden=False)
us_trajectory = trajectory_radial[0::5]
us_img = hrecon.undersample_img(A_img, us_trajectory)

fig, ax = plt.subplots(3, 4)
fig2, ax2 = plt.subplots(3, 4)
ax = ax.ravel()
ax2 = ax2.ravel()
for i, i_ax in enumerate(ax):
    z_abs = np.abs(us_img)[-i]
    z_angle = np.angle(us_img)[-i]
    # z_abs = np.abs(np.fft.fft2(us_img, norm='ortho')[-i]).ravel()
    # z = harray.scale_minmax(z)
    # z = harray.scale_minpercentile_both(z, q=95)
    # z_max = np.percentile(np.abs(z), 99)
    # z = z / z_max
    # i_ax.hist(z, range=(0 + 1e-3, 1), bins=256)
    i_ax.imshow(z_abs)
    ax2[i].imshow(z_angle)

fig.savefig('/home/sharreve/synth_us_abs.png')
fig2.savefig('/home/sharreve/synth_us_angle.png')


# # # Example inference data
ddata_inf = os.path.join(DINFERENCE, 'input')
file_list = os.listdir(ddata_inf)
sel_file = file_list[0]
file_path = os.path.join(ddata_inf, sel_file)
B = hmisc.load_array(file_path, data_key='kspace', sel_slice='mid')
B_cpx = B[..., ::2] + 1j * B[..., 1::2]
B_cpx = np.moveaxis(B_cpx, -1, 0)
# B_img = np.fft.ifft2(np.fft.fftshift(B_cpx, axes=(-2, -1)), norm='ortho')
B_img = np.fft.ifft2(B_cpx, norm='ortho')
print('Shape of cpx', B_cpx.shape)

B_sos = hrecon.convert_to_sos(B)
fig_obj = hplotc.ListPlot([B_sos], cbar=True)
fig_obj.figure.savefig(os.path.expanduser('~/test_target_inf_sos.png'))



fig, ax = plt.subplots(3,4)
fig2, ax2 = plt.subplots(3,4)
ax = ax.ravel()
ax2 = ax2.ravel()
for i, i_ax in enumerate(ax):
    z_abs = np.abs(B_img)[-i]
    z_angle = np.angle(B_img)[-i]
    # z_abs = np.abs(np.fft.fft2(us_img, norm='ortho')[-i]).ravel()
    # z = harray.scale_minmax(z)
    # z = harray.scale_minpercentile_both(z, q=95)
    # z_max = np.percentile(np.abs(z), 99)
    # z = z / z_max
    # i_ax.hist(z, range=(0 + 1e-3, 1), bins=256)
    i_ax.imshow(z_abs)
    ax2[i].imshow(z_angle)

fig.savefig('/home/sharreve/us_abs.png')
fig2.savefig('/home/sharreve/us_angle.png')
#
# fig, ax = plt.subplots(3,4)
# ax = ax.ravel()
# for i, i_ax in enumerate(ax):
#     # z = np.abs(B_img[-i]).ravel()
#     z = np.abs(np.fft.fft2(B_img, norm='ortho')[-i]).ravel()
#     z = harray.scale_minmax(z)
#     i_ax.hist(z, range=(0 + 1e-1, 1), bins=256)

# fig.savefig('/home/sharreve/hist_us.png')


# Store the SoS
A_sos = hrecon.convert_to_sos(A)
B_sos = hrecon.convert_to_sos(B)
C_sos = np.sqrt((np.abs(us_img[-12:]) ** 2).sum(axis=0))
# harray.get_minmeanmediammax(A_sos)
# harray.get_minmeanmediammax(B_sos)
# harray.get_minmeanmediammax(C_sos)

# # # Plot these..
fig_obj = hplotc.ListPlot([A_sos, B_sos, C_sos], cbar=True)
fig_obj.figure.savefig(os.path.expanduser('~/test.png'))


"""
Any how does the Calgary Dataset look like...?

"""

import helper.plot_class as hplotc
import helper.misc as hmisc
from objective_helper.reconstruction import circus_radial_mask
import numpy as np
import matplotlib.pyplot as plt

dd = '/media/bugger/MyBook/data/calgarycampinas/conp-dataset/projects/calgary-campinas/CC359/Raw-data/Multi-channel/12-channel/test_12_channel/Test-R=5'
import os
file_list = os.listdir(dd)

sel_file = os.path.join(dd, file_list[10])

A = hmisc.load_array(sel_file, data_key='kspace', sel_slice='mid')
A_cpx = A[..., ::2] + 1j * A[..., ::2]
A_cpx = np.moveaxis(A_cpx, -1, 0)
# A_cpx = np.pad(A_cpx, ((0,0), (0,0), (24, 24)))
sel_coil = A_cpx[0]

ifft2 = np.fft.fft2(sel_coil)
ifftshift = np.fft.ifftshift(sel_coil)
fftshift = np.fft.fftshift(sel_coil)
ifft2_ifftshift = np.fft.ifft2(ifftshift)
ifft2_fftshift = np.fft.ifft2(fftshift)

hplotc.ListPlot([sel_coil, ifft2, ifftshift, fftshift, ifft2_ifftshift, ifft2_fftshift])