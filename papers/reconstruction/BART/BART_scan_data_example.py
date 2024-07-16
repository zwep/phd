import bart
import helper.reconstruction as hrecon
import scipy.ndimage
import helper.array_transf as harray
from reconstruction import ReadCpx
import helper.misc as hmisc
import helper.plot_class as hplotc
import os
import cfl
import matplotlib.pyplot as plt
import numpy as np
from objective_configuration.reconstruction import DPLOT
from objective_helper.reconstruction import visualize_traj_BART, nufft_rss, nufft_pics
import objective_helper.reconstruction as obj_helper

dplot = os.path.join(DPLOT, 'cardiac_scan_nufft')
if not os.path.isdir(dplot):
    os.makedirs(dplot)

"""
Load a cartesian sampled complex valued image file -> regrid to spokes -> nufft with BART
"""


sel_cpx_array, kspace_array = obj_helper.get_cartesian_sampled_cardiac_data()
nkyz, _ = sel_cpx_array.shape

# Visualize the image we took
fig_obj_7T = hplotc.ListPlot([np.abs(sel_cpx_array), np.angle(sel_cpx_array)], col_row=(2,1), title='Example of 7T image')
fig_obj_7T.figure.savefig(os.path.join(dplot,'example 7T image.png'))

n_spokes = nkyz // 4
traj_rad = bart.bart(1, f'traj -r -x{nkyz} -y{n_spokes}')

# Now try to (re)grid this....
traj_rad_real = traj_rad.real - traj_rad.min().real
radial_kspace_grid = scipy.ndimage.map_coordinates(kspace_array, traj_rad_real[:2])
fig_obj_regrid = hplotc.ListPlot(radial_kspace_grid, augm='np.abs')
fig_obj_regrid.figure.savefig(os.path.join(dplot, 'regrid of 7T image.png'))

res = nufft_rss(traj_rad, radial_kspace_grid[None])
res = harray.scale_minmax(res, is_complex=True)
sel_cpx_array = harray.scale_minmax(sel_cpx_array, is_complex=True)

fig_obj_result = hplotc.ListPlot([res, sel_cpx_array, np.abs(res) - np.abs(sel_cpx_array)], augm='np.abs',
                title='Check effect of regridding without SENSE file', cbar=True, ax_off=True,
                subtitle=[['Result from BART'], ['Original'], ['Difference']], col_row=(3, 1))
fig_obj_result.figure.savefig(os.path.join(dplot, 'result undersampled nufft.png'))

"""
Load a radial sampled images and solve it with PI-CS and with an undersampled nufft
"""

# Get the radial sampled data
sel_cpx_array = obj_helper.get_radial_sampled_cardiac_data()
abs_sos = np.sqrt((np.abs(sel_cpx_array)**2).sum(axis=0))

# Get the sensitivity map (approximation...)
n_size = sel_cpx_array.shape[-1]
sense_img = obj_helper.get_sensitivity_map_radial_sampled_cardiac_data(n_size)

# Regrid the radial data to specific spokes...
kspace_coil_list = [harray.transform_image_to_kspace_fftn(x) for x in sel_cpx_array]
nkyz = sel_cpx_array.shape[-1]

# Reduce the number of spokes we work with...
n_spokes = nkyz // 4
traj_rad = bart.bart(1, f'traj -r -x{nkyz} -y{n_spokes}')
traj_rad_real = traj_rad.real - traj_rad.min().real
radial_kspace_grid_list = [scipy.ndimage.map_coordinates(x, traj_rad_real[:2]) for x in kspace_coil_list]


# Format the dimensions so that BART can accept it...
sense_img_bart = np.expand_dims(np.moveaxis(sense_img, 0, -1), 2)
radial_data_bart = np.stack(radial_kspace_grid_list, axis=-1)[None]

# Direct NUFFT
reco1 = nufft_rss(traj_rad, radial_data_bart)
# With some estimated sensitivity map...
# I know that the sense map is crappy, but it is something.
reco2 = bart.bart(1, 'pics -S -r0.1 -t', traj_rad, radial_data_bart, sense_img_bart)

# Also try it with L1 wavelet stuff: no sensible solutions
list_of_l1wave_param = [x.round(2) for x in np.arange(0, 1, 0.1)]
nrow, ncol = hmisc.get_square(len(list_of_l1wave_param))
im_l1wave_list = []
for l1wave_reg in list_of_l1wave_param:
    im_wav = bart.bart(1, 'pics -R W:6:0:{} -S -d5 -t'.format(l1wave_reg), traj_rad, radial_data_bart, sense_img_bart)
    im_l1wave_list.append(im_wav)

hplotc.ListPlot(im_l1wave_list, augm='np.abs')

# Last thing, but now with Total Variation: hardly any effect
list_of_TV_param = [x.round(2) for x in np.arange(0, 1, 0.1)]
nrow, ncol = hmisc.get_square(len(list_of_TV_param))
im_TV_list = []
for TV_reg in list_of_TV_param:
    im_wav = bart.bart(1, 'pics -R T:6:0:{} -S -d5 -t'.format(TV_reg), traj_rad, radial_data_bart, sense_img_bart)
    im_TV_list.append(im_wav)

hplotc.SlidingPlot(np.array(im_TV_list))


hplotc.ListPlot([reco1, reco2], augm='np.abs')