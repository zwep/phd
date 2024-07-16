import os
import scipy
import helper.misc as hmisc
import numpy as np
import torchaudio
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.reconstruction as hrecon
import sigpy
import reconstruction.ReadCpx as read_cpx

import matplotlib.pyplot as plt
import skimage.data

"""

check the spectogram of some radial sampled image...

"""

ddata = '/media/bugger/MyBook/data/7T_data/cardiac_radial_us_fs/input/v9_02052021_1432554_V9_19531_trans.h5'
single_img = skimage.data.astronaut()[:, :, 0].astype(float)

# filename = 'v9_31012021_1547133_5_3_surveylrV4'
# read_cpx_obj = read_cpx.ReadCpx(os.path.join(ddata, filename))
# image_array = read_cpx_obj.read_cpx_img()

kspace_array = hmisc.load_array(ddata, data_key='kspace')
kspace_array_cpx = kspace_array[..., ::2] + 1j * kspace_array[..., 1::2]
kspace_array_cpx = np.moveaxis(kspace_array_cpx, -1, 0)
sel_coil_index = 4
sel_loc = 0
single_kspace = kspace_array_cpx[sel_coil_index, sel_loc]
single_img = np.fft.ifftn(np.fft.fftshift(single_kspace), axes=(0, 1))
# single_img = np.fft.ifftn(single_kspace, axes=(0, 1))

single_kspace = np.fft.ifft2(single_img)
# Represent the image
hplotc.ListPlot([single_img, single_kspace])

trajectory_radial = hrecon.get_trajectory(single_img.shape)
max_spokes, n_points, _ = trajectory_radial.shape
trajectory_radial = trajectory_radial.reshape(-1, 2)
dcf = np.sqrt(trajectory_radial[:, 0] ** 2 + trajectory_radial[:, 1] ** 2)

z = hrecon.get_trajectory_n_spoke((-128, -128, 0), (128, 128, 0))
# display the traj
# z = trajectory_radial.reshape((max_spokes, n_points, 2))
for ii in range(0, 256, 11):
    plt.scatter(z[ii, :, 0], z[ii, :, 1])

temp_kspace = sigpy.nufft(single_img, coord=z, width=6, oversamp=1.25)
a = temp_kspace.reshape((256, 256))
hplotc.ListPlot(a)


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import square, ShortTimeFFT
from scipy.signal.windows import gaussian

g_std = 12  # standard deviation for Gaussian window in samples
win = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian wind.

stfft_obj = scipy.signal.ShortTimeFFT(win=win, hop=1, fs=1/256, fft_mode='twosided', mfft=350)
res = []
for ia in a:
    z = stfft_obj.spectrogram(ia)
    z_dB = 10 * np.log10(np.fmax(z, 1e-4))  # limit range to -40 dB
    res.append(z_dB)

res_array = np.array(res)
res_array.shape
hplotc.SlidingPlot(res_array)

T_x, N = 1 / 20, 1000  # 20 Hz sampling rate for 50 s signal
t_x = np.arange(N) * T_x  # time indexes for signal
f_i = 5e-3*(t_x - t_x[N // 3])**2 + 1  # varying frequency
x = square(2*np.pi*np.cumsum(f_i)*T_x)  # the signal

# The utitlized Gaussian window is 50 samples or 2.5 s long. The
# parameter ``mfft=800`` (oversampling factor 16) and the `hop` interval
# of 2 in `ShortTimeFFT` was chosen to produce a sufficient number of
# points:

g_std = 12  # standard deviation for Gaussian window in samples
win = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian wind.
SFT = ShortTimeFFT(win, hop=2, fs=1/T_x, scale_to='psd')
Sx2 = SFT.spectrogram(x)  # calculate absolute square of STFT
Sx2.shape

# The plot's colormap is logarithmically scaled as the power spectral
# density is in dB. The time extent of the signal `x` is marked by
# vertical dashed lines and the shaded areas mark the presence of border
# effects:

fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
ax1.set_title(rf"Spectrogram ({SFT.m_num*SFT.T:g}$\,s$ Gaussian " + rf"window, $\sigma_t={g_std*SFT.T:g}\,$s)")
ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +  rf"$\Delta t = {SFT.delta_t:g}\,$s)",
 ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +  rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
 xlim=(t_lo, t_hi))
Sx_dB = 10 * np.log10(np.fmax(Sx2, 1e-4))  # limit range to -40 dB
im1 = ax1.imshow(Sx_dB, origin='lower', aspect='auto',
          extent=SFT.extent(N), cmap='magma')
ax1.plot(t_x, f_i, 'g--', alpha=.5, label='$f_i(t)$')
fig1.colorbar(im1, label='Power Spectral Density ' +
                  r"$20\,\log_{10}|S_x(t, f)|$ in dB")
...
# Shade areas where window slices stick out to the side:
for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
          (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
...     ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.3)
for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line
...     ax1.axvline(t_, color='c', linestyle='--', alpha=0.5)
ax1.legend()
fig1.tight_layout()
plt.show()
