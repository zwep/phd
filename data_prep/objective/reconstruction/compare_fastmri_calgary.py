import os
import helper.plot_class as hplotc
import h5py
import numpy as np
import helper.misc as hmisc
import helper.array_transf as harray

"""
They say that I need a centered fft for FastMRI data and non-centered for Calgary

Can I visualize this for myself and confirm one or the other..?
"""

DFASTMRI_knee = '/nfs/arch11/researchData/PROJECT/fastMRI_NYU/knee/multicoil_train'
DSEB_cardiac = '/local_scratch/sharreve/mri_data/cardiac/test'
DCALGARY_brain = '/nfs/arch11/researchData/PROJECT/fastMRI_NYU/CC359/Raw-data/Multi-channel/12-channel/Test-R=5'
DPLOT = '/local_scratch/sharreve/paper/reconstruction/results/plot'


def select_and_load(ddir, data_source):
    file_list = os.listdir(ddir)
    file_list = [x for x in file_list if x.endswith('h5')]
    sel_file = file_list[0]
    sel_file_path = os.path.join(ddir, sel_file)
    with h5py.File(sel_file_path, 'r') as f:
        print(f['kspace'].shape)
        if data_source == 'nyu':
            # (35, 15, 640, 372)
            nloc, ncoil, ny, nx = f['kspace'].shape
            sel_array = np.array(f['kspace'][nloc//2])
        elif data_source == 'calgary':
            # (25, 401, 401, 48)
            nloc, ny, nx, ncoil = f['kspace'].shape
            sel_array = np.array(f['kspace'][nloc // 2])
            sel_array = np.moveaxis(sel_array, -1, 0)
    return sel_array


array_nyu = select_and_load(DFASTMRI_knee, data_source='nyu')
array_calgary = select_and_load(DCALGARY_brain, data_source='calgary')
# array_cardiac = select_and_load(DSEB_cardiac, data_source='calgary')

array_calgary_cpx = array_calgary[::2] + 1j * array_calgary[1::2]
# array_cardiac_cpx = array_cardiac[::2] + 1j * array_cardiac[1::2]

array_cc_split = np.stack([array_calgary[::2], array_calgary[1::2]], axis=-1)

"""
NYU
"""

# Check the kspace data from NYU
dest_png = os.path.join(DPLOT, 'knee_nyu.png')
nyu_max = np.abs(array_nyu[0]).max()
nyu_min = np.abs(array_nyu[0]).min()
fig_obj = hplotc.ListPlot(array_nyu[0], augm='np.abs', vmin=(0.8 * nyu_min, 0.8 * nyu_max))
fig_obj.figure.savefig(dest_png)

# Check what FFT transform does when using a center transform..
dest_png = os.path.join(DPLOT, 'knee_nyu_img.png')
nyu_img_space = [harray.transform_kspace_to_image_fftn(x) for x in array_nyu]
fig_obj = hplotc.ListPlot(nyu_img_space, augm='np.abs', col_row=(4, 4))
fig_obj.figure.savefig(dest_png)

"""
Calgaray
"""

# # Test ifft2 on Calgary
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
abs_cc = np.abs(array_calgary_cpx[0])
ax.imshow(abs_cc, vmin=(0, 0.2 * np.max(abs_cc)))
fig.savefig('/home/sharreve/test_cc.png')

import torch
fig, ax = plt.subplots()
abs_cc_split = ifft2(torch.from_numpy(array_cc_split[0:1]), centered=False).numpy()
abs_cc = np.real(abs_cc_split[0, ..., 0] + 1j * abs_cc_split[0, ..., 1])
ax.imshow(abs_cc, vmin=(0, 0.2 * np.max(abs_cc)))
fig.savefig('/home/sharreve/test_cc_img.png')

# Check the kspace data from Calgary
fig_obj = hplotc.ListPlot(array_calgary_cpx, col_row=(3, 4), augm='np.abs', cbar=True, vmin=(0, 0.2 * np.abs(array_calgary_cpx).max()))
fig_obj.figure.savefig(os.path.join(DPLOT, 'brain_calgary.png'))

# Check what FFT transform does when using a center transform..
calgary_img_space = [harray.transform_kspace_to_image_fftn(x) for x in array_calgary_cpx]
fig_obj = hplotc.ListPlot(calgary_img_space, augm='np.abs', col_row=(4, 4))
fig_obj.figure.savefig(os.path.join(DPLOT, 'brain_calgary_img.png'))

calgary_img_space = [np.fft.ifft2(x) for x in array_calgary_cpx]
fig_obj = hplotc.ListPlot(calgary_img_space, augm='np.real', col_row=(4, 4))
fig_obj.figure.savefig(os.path.join(DPLOT, 'brain_calgary_img_fft.png'))

calgary_img_space = [np.fft.ifft2(np.fft.ifftshift(x)) for x in array_calgary_cpx]
fig_obj = hplotc.ListPlot(calgary_img_space, augm='np.real', col_row=(4, 4))
fig_obj.figure.savefig(os.path.join(DPLOT, 'brain_calgary_img_fft_shift.png'))

calgary_img_space = [np.fft.ifftshift(x) for x in array_calgary_cpx]
fig_obj = hplotc.ListPlot(calgary_img_space, augm='np.abs', col_row=(4, 4), vmin=(0, 0.1 * np.abs(array_calgary_cpx).max()))
fig_obj.figure.savefig(os.path.join(DPLOT, 'brain_calgary_img_shift.png'))



"""
Seb
"""
#
# # Check the kspace data from Calgary
# fig_obj = hplotc.ListPlot(array_cardiac_cpx[0], augm='np.abs', cbar=True, vmin=(0, 0.2 * np.abs(array_cardiac_cpx).max()))
# fig_obj.figure.savefig(os.path.join(DPLOT, 'cardiac_seb.png'))
#
# # Check what FFT transform does when using a center transform..
# cardiacary_img_space = [harray.transform_kspace_to_image_fftn(x) for x in array_cardiac_cpx]
# fig_obj = hplotc.ListPlot(cardiacary_img_space, augm='np.abs', col_row=(4, 4))
# fig_obj.figure.savefig(os.path.join(DPLOT, 'cardiac_seb_img.png'))
#
# # This worked before..
# A_sos = np.sqrt(np.sum(np.abs(harray.transform_kspace_to_image_fftn(array_cardiac_cpx, dim=(-2, -1))) ** 2, axis=0))
# fig_obj = hplotc.ListPlot(A_sos, augm='np.abs')
# fig_obj.figure.savefig(os.path.join(DPLOT, 'cardiac_seb_sos.png'))
# A_fft = harray.transform_kspace_to_image_fftn(array_cardiac_cpx[-1], dim=(-2, -1))
# fig_obj = hplotc.ListPlot(A_fft, augm='np.abs')
# fig_obj.figure.savefig(os.path.join(DPLOT, 'test_fft.png'))
# """
# """

from objective_helper.reconstruction import step_by_step_plot

fig_obj = step_by_step_plot(sel_coil=array_calgary_cpx[0])
name = 'step_by_step_calgary_abs.png'
fig_obj.figure.savefig(os.path.join(DPLOT, name))
# step_by_step_plot(sel_coil=array_cardiac_cpx[-1], name='step_by_step_cardiac_abs.png')

fig_obj = step_by_step_plot(sel_coil=array_calgary_cpx[0], operator='np.angle')
name = 'step_by_step_calgary_angle.png'
fig_obj.figure.savefig(os.path.join(DPLOT, name))
# step_by_step_plot(sel_coil=array_cardiac_cpx[-1], name='step_by_step_cardiac_angle.png', operator='np.angle')

# np.sum(array_calgary_cpx[0] == array_calgary_cpx[0].T)
# np.sum(array_cardiac_cpx[0] == array_cardiac_cpx[0].conjugate().T)
