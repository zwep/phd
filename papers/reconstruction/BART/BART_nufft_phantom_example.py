import bart
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

"""
Adapted version of the DEMO page...
"""

dplot = os.path.join(DPLOT, 'phantom_data_nufft')
if not os.path.isdir(dplot):
    os.makedirs(dplot)

"""
Below is an example of nufft on phantom data where we try to reconstruction techniques

1. Direct NUFFT
2. Compressed sense and parallel imaging (PICS)
"""

N_SPOKES = 64

# Generate k-space trajectory with 64 radial spokes
traj_rad = bart.bart(1, f'traj -r -x512 -y{N_SPOKES}')

# 2x oversampling
traj_rad_scaled = bart.bart(1, 'scale 0.5', traj_rad)

# simulate eight-channel k-space data
ksp_sim = bart.bart(1, 'phantom -k -s8 -t', traj_rad_scaled)

# increase the reconstructed FOV a bit
reco1 = nufft_rss(traj_rad_scaled, ksp_sim)
# visualize_traj_BART(traj_temp[:, :, ::4])
vmax = 0.4 * int(np.abs(reco1.max()))
fig_obj_recon1 = hplotc.ListPlot([reco1], augm='np.abs', vmin=(0, vmax), title=f'Direct NUFFT with {N_SPOKES} spokes')
fig_obj_recon1.figure.savefig(os.path.join(dplot, 'direct nufft.png'))

# Try different undersampling patterns
recon_list = [nufft_rss(traj_rad_scaled[:, :, ::ii], ksp_sim[:, :, ::ii]) for ii in [2, 4, 6]]
fig_obj_recon_list = hplotc.ListPlot(recon_list, augm='np.abs', title='Varying number of spokes by a factor of 2, 4 and 6')
fig_obj_recon_list.figure.savefig(os.path.join(dplot, 'varying number of spokes nufft.png'))

# Do some pre-processing steps to create a sensitivity-map
#   reconstruct low-resolution image and transform back to k-space
lowres_img = bart.bart(1, 'nufft -i -d24:24:1 -t', traj_rad_scaled, ksp_sim)
lowres_ksp = bart.bart(1, 'fft -u 7', lowres_img)

#   zeropad to full size
ksp_zerop = bart.bart(1, 'resize -c 0 308 1 308', lowres_ksp)
# ksp_zerop.shape: (308, 308, 1, 8)

#   ESPIRiT calibration
sens = bart.bart(1, 'ecalib -m1', ksp_zerop)
# sens.shape: (308, 308, 1, 8)

#   Visualize the sensitivity patterns
fig_obj_sens = hplotc.ListPlot([sens.T], augm='np.abs', ax_off=True)
fig_obj_sens.figure.savefig(os.path.join(dplot, 'estimated sensitivity profiles.png'))

# ksp_sim.shape: (1, 512, 64, 8)
# traj_rad_scaled.shape: (3, 512, 64)

# Now we can apply PICS
# Non-Cartesian parallel imaging
reco2 = bart.bart(1, 'pics -S -r0.001 -t', traj_rad_scaled, ksp_sim, sens)
vmax = 0.4 * int(np.abs(reco2.max()))
fig_obj_recon2 = hplotc.ListPlot([reco2], augm='np.abs', vmin=(0, vmax), title=f'PI-CS with {N_SPOKES} spokes')
fig_obj_recon2.figure.savefig(os.path.join(dplot, 'PI-CS.png'))

# Display different undersampling techniques
recon_pics_list = [nufft_pics(traj_rad_scaled[:, :, ::ii], ksp_sim[:, :, ::ii], sens) for ii in [2, 4, 6]]
fig_obj_pics_list = hplotc.ListPlot(recon_pics_list, augm='np.abs', title='varying number of spokes pi-cs.png')
fig_obj_pics_list.figure.savefig(os.path.join(dplot, 'varying number of spokes PI-CS.png'))

