import bart
import helper.misc as hmisc
import helper.plot_class as hplotc
import os
import cfl
import matplotlib.pyplot as plt
import numpy as np

from objective_configuration.reconstruction import DPLOT

"""
There is a demo online

https://github.com/mrirecon/bart-webinars/blob/master/webinar4/demo/demo.ipynb

I am simply copying the code, and maybe play around a bit...
"""

dplot = os.path.join(DPLOT, 'phantom_data_cartesian')
if not os.path.isdir(dplot):
    os.makedirs(dplot)


# Create phantom data
# This can easily be swapped with my data
nkyz = 256
nc = 8
ksp_ful_nonoise = bart.bart(1, 'phantom -x {} -s {} -k'.format(nkyz, nc))
ksp_ful_nonoise.shape = (1, nkyz, nkyz, nc)

# Create noise
noi_var = 1000
seed = 20210714
ksp_ful = bart.bart(1, 'noise -n {} -s {}'.format(noi_var, seed), ksp_ful_nonoise)
# This is needed when we only simulate one coil..
if ksp_ful.ndim == 3:
    ksp_ful = ksp_ful[:, :, :, None]

# Create a sampling mask
usamp_lvl_y = 3
usamp_lvl_z = 1.5
calib_dim = 16
vdf = 20
mask_args = (nkyz, nkyz, usamp_lvl_y, usamp_lvl_z, calib_dim, vdf)
mask = bart.bart(1, 'poisson -Y {} -Z {} -y {} -z {} -C {} -V {} -e'.format(*mask_args))
mask.shape = (1, nkyz, nkyz, 1)  # for array broadcasting

# Display mask and kspace
ksp = ksp_ful * mask
max_value = np.abs(ksp).max()
fig_obj = hplotc.ListPlot(ksp[0].T, augm='np.abs', col_row=(2, 4), vmin=(0, 0.01 * max_value), title='Sampling masks')
fig_obj.figure.savefig(os.path.join(dplot, fig_obj.title_string + '.png'))


# Pycharm cant execute this? !bart bitmask 1 2
# Pretty weird, because the in-built shell can...
# The result of this is '6'
# This bitmask is needed to know along which dimensions to fft!
im_coil = bart.bart(1, 'fft -u -i 6', ksp)
fig_obj = hplotc.ListPlot(im_coil[0].T, augm='np.abs', col_row=(2, 4), vmin=(0, 0.01 * max_value), title='Zero-filled FFT')
fig_obj.figure.savefig(os.path.join(dplot, fig_obj.title_string + '.png'))


# Here 8 is the result of bart bitmask 1 2
im_rss = bart.bart(1, 'rss 8', im_coil)
fig_obj = hplotc.ListPlot(im_rss[0], augm='np.abs', title='Root Sum of Squares image')
fig_obj.figure.savefig(os.path.join(dplot, fig_obj.title_string + '.png'))

# Now try it with estiamte coil sensitivities..
ksp_smap_nonoise = bart.bart(1, 'phantom -x {} -k -s {}'.format(calib_dim, nc))
ksp_smap_nonoise.shape = (1, calib_dim, calib_dim, nc)
seed_smap = 14072021
ksp_smap = bart.bart(1, 'noise -n {} -s {}'.format(noi_var, seed_smap), ksp_smap_nonoise)
ksp_smap_pad = bart.bart(1, 'resize -c 1 {} 2 {}'.format(nkyz, nkyz), ksp_smap)

# Do stuff... CC is some compression operator that can be applied to kspace map map and kspace
cc_mtx = bart.bart(1, 'cc -S -M', ksp_smap_pad)
nvc = 5
ksp_smap_cc = bart.bart(1, 'ccapply -p {} -S'.format(nvc), ksp_smap_pad, cc_mtx)
ksp_cc = bart.bart(1, 'ccapply -p {} -S'.format(nvc), ksp, cc_mtx)

# Sensitivity maps...
smap = bart.bart(1, 'ecalib -m 1 -S', ksp_smap_cc)
fig_obj = hplotc.ListPlot(smap[0].T, augm='np.abs', title='Estimated sensitivity maps')
fig_obj.figure.savefig(os.path.join(dplot, fig_obj.title_string + '.png'))

# Now we can do iterative reconstruction!
# Here are some solutions with a range of regularizations
list_of_l2_param = [x.round(2) for x in np.arange(0, 1, 0.1)]
nrow, ncol = hmisc.get_square(len(list_of_l2_param))
im_l2_list = []
for l2_reg in list_of_l2_param:
    im_l2 = bart.bart(1, 'pics -R Q:{} -S -d5'.format(l2_reg), ksp_cc, smap)
    im_l2_list.append(im_l2)

fig_obj = hplotc.ListPlot(im_l2_list, augm='np.abs', title='L2-regularized solution', col_row=(nrow, ncol),
                          ax_off=True, subtitle=[[x] for x in list_of_l2_param])
fig_obj.figure.savefig(os.path.join(dplot, fig_obj.title_string + '.png'))

# Same thing, now L1-regularization
list_of_l1wave_param = [x.round(2) for x in np.arange(0, 1, 0.1)]
nrow, ncol = hmisc.get_square(len(list_of_l1wave_param))
im_l1wave_list = []
for l1wave_reg in list_of_l1wave_param:
    im_wav = bart.bart(1, 'pics -R W:6:0:{} -S -d5'.format(l1wave_reg), ksp_cc, smap)
    im_l1wave_list.append(im_wav)

fig_obj = hplotc.ListPlot(im_l1wave_list, augm='np.abs', title='L1-wavelet-regularized solution', col_row=(nrow, ncol), ax_off=True,
                subtitle=[[x] for x in list_of_l1wave_param])
fig_obj.figure.savefig(os.path.join(dplot, fig_obj.title_string + '.png'))

# Last thing, but now with Total Variation
list_of_TV_param = [x.round(2) for x in np.arange(0, 1, 0.1)]
nrow, ncol = hmisc.get_square(len(list_of_TV_param))
im_TV_list = []
for TV_reg in list_of_TV_param:
    im_wav = bart.bart(1, 'pics -R T:6:0:{} -S -d5'.format(TV_reg), ksp_cc, smap)
    im_TV_list.append(im_wav)

fig_obj = hplotc.ListPlot(im_TV_list, augm='np.abs', title='TV-regularized solution', col_row=(nrow, ncol), ax_off=True,
                subtitle=[[x] for x in list_of_TV_param])
fig_obj.figure.savefig(os.path.join(dplot, fig_obj.title_string + '.png'))
