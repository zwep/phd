from data_generator.Reconstruction import DataGeneratorReconstruction
import helper.array_transf as harray
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import matplotlib.pyplot as plt
import helper.misc as hmisc
import numpy as np
import os
import helper.plot_class as hplotc

"""
We are going to check how us differs from synth u.s.

--> Currently this is on hold since the inference data that I have on my HDD contains a checkerboard pattern.
This can severaly impact this study

Is this also online? Both for inference as well as training data?
Lets find it out

After we have downloaded all the sim SA data from MM1A
"""


def direct2cpx(file_path, data_key='data'):
    kspace = hmisc.load_array(file_path, data_key=data_key)
    kspace_cpx = kspace[..., ::2] + 1j * kspace[..., 1::2]  # Convert real-valued to complex-valued data.
    # For a proper real-channel thing this is needed
    kspace_cpx = np.fft.fftshift(kspace_cpx, axes=(-3, -2))
    imgspace_cpx = np.fft.fftn(kspace_cpx, axes=(-3, -2))
    return imgspace_cpx, kspace_cpx


def plot_hist(x_array, label_array):
    n_plot = len(label_array)
    plot_size = hmisc.get_square(n_plot)
    fig, ax = plt.subplots(*plot_size)
    ax = ax.ravel()
    for i, (x, ilabel) in enumerate(zip(x_array, label_array)):
        ax[i].hist(x[x != 0].ravel(), label=ilabel, bins=256)
        ax[i].legend()
    return fig, ax


ddata = '/media/bugger/MyBook/data/7T_data/cardiac_radial_us_fs'
datagen_obj = DataGeneratorReconstruction(ddata, dataset_type='', file_ext='h5', switch_input_output=True)

ddata_us = os.path.join(ddata, 'input')
ddata_fs = os.path.join(ddata, 'target')
file_list = os.listdir(ddata_fs)
sel_card = 0

collect_us_noise = []
collect_synth_us_noise = []

i_file = file_list[0]
for i_file in file_list:
    us_file = os.path.join(ddata_us, i_file)
    fs_file = os.path.join(ddata_fs, i_file)
    us_img, us_kspace = direct2cpx(us_file)
    fs_img, fs_kspace = direct2cpx(fs_file)
    n_card = fs_kspace.shape[0]
    n_card_us = us_kspace.shape[0]
    # i_card zero is the only reliable position that the hearts are the same
    sel_card = 0
    fs_kspace_moved = np.moveaxis(fs_kspace[sel_card], -1, 0)
    fs_img_moved = np.moveaxis(fs_img[sel_card], -1, 0)
    us_img_moved = np.moveaxis(us_img[sel_card], -1, 0)
    # Haha now I need to re-shift this again... because..
    fs_kspace_moved = np.fft.fftshift(fs_kspace_moved, axes=(-2, -1))
    synth_us_img, _ = datagen_obj.undersample_image(fs_kspace_moved)
    synth_us_img = synth_us_img[:, ::-1, ::-1]

    circular_mask = hmisc.circular_mask(fs_img_moved.shape[-2:])

    synth_us_img = harray.scale_minmax(synth_us_img * circular_mask, is_complex=True)
    fs_img_moved = harray.scale_minmax(fs_img_moved * circular_mask, is_complex=True)
    us_img_moved = harray.scale_minmax(us_img_moved * circular_mask, is_complex=True)

    # Checking the images...
    sel_coil = 0
    hplotc.ListPlot([us_img_moved[sel_coil], fs_img_moved[sel_coil], synth_us_img[sel_coil]], augm='np.real')
    n_coil = us_img_moved.shape[0]
    for i_coil in range(n_coil):
        plot_array = [np.real(x) for x in [synth_us_img[i_coil], fs_img_moved[i_coil], us_img_moved[i_coil]]]
        patch_size = 64
        vmax_list = [(-harray.get_proper_scaled_v2(-x, (patch_size, patch_size), patch_size // 2), harray.get_proper_scaled_v2(x, (patch_size, patch_size), patch_size // 2)) for x in plot_array]
        hplotc.ListPlot(np.array(plot_array)[None], ax_off=True, hspace=0, wspace=0, vmin=[vmax_list], figsize=(30, 10))

        plot_array_tresh = [harray.treshold_image(x, x_tresh[-1]) for x, x_tresh in zip(plot_array, vmax_list)]

        synth_us_difference = plot_array_tresh[0] - plot_array_tresh[1]
        us_difference = plot_array_tresh[-1] - plot_array_tresh[1]

        synth_ravel = np.real(synth_us_difference.ravel())
        collect_synth_us_noise.extend(synth_ravel)
        us_ravel = np.real(us_difference.ravel())
        collect_us_noise.extend(us_ravel)


for idata in [collect_us_noise, collect_synth_us_noise]:
    distributions = [stats.norm, stats.expon, stats.gamma]
    data = np.array(idata)
    best_fit = None
    best_params = None
    best_distribution = None
    best_sse = np.inf

    for distribution in distributions:
        # Fit the distribution to the data
        params = distribution.fit(data)

        # Calculate the PDF and SSE (sum of squared errors)
        pdf = distribution.pdf(data, *params)
        sse = np.sum((pdf - data) ** 2)

        # Check if this fit is better than the previous best fit
        if sse < best_sse:
            best_fit = distribution
            best_params = params
            best_distribution = distribution.name
            best_sse = sse

    print(f"Best distribution: {best_distribution}")
    print(f"Best parameters: {best_params}")

# # Plot the best-fitting distribution
x = np.linspace(min(data), max(data), 1000)
pdf = stats.expon.pdf(x, 0.6) -0.6
plt.plot(x, pdf, 'k-', label=f'Best fit ({best_distribution})')
# # plt.hist(data)
# # Display the plot
# plt.legend()
# plt.show()
#
