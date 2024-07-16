
"""
HUM - Homomorphic Unsharp Masking

Ardizzone, Edoardo, Roberto Pirrone, and Orazio Gambino. "Bias Correction on Knee MR Images."
IADAT-micv2005
International Conference on Multimedia, Image Processing and Computer Vision. 2005.
"""

import numpy as np

import helper.plot_fun as hplotf
import nibabel
import helper.array_transf as harray
import scipy.signal
import helper.plot_class as hplotc


def get_filter(nx, ny, d0, d1, n):
    filter1 = np.ones((nx, ny))
    filter2 = np.ones((nx, ny))
    filter3 = np.ones((nx, ny))
    for i in range(nx):
        for j in range(ny):
            dist = ((i-(nx/2)) ** 2 + (j-(ny/2)) ** 2) ** 0.5
            # No idea.//
            if dist == 0:
                dist = 1
            filter1[i, j] = 1/(1 + (dist/d1) ** (2*n))
            filter2[i, j] = 1/(1 + (dist/d0) ** (2*n))
            filter3[i, j] = 1.0 - filter2[i, j]
            filter3[i, j] = filter1[i, j] * filter3[i, j]

    return filter3


def butterworthbpf(I, d0, d1, n, debug=False):
    nx, ny = I.shape
    I_fft = harray.transform_image_to_kspace_fftn(I)

    filter3 = get_filter(nx, ny, d0, d1, n)
    # filtered_image = I_fft * (1 + filter3)
    filtered_image = I_fft * filter3
    filtered_image = harray.transform_kspace_to_image_fftn(filtered_image)
    if debug:
        return filtered_image, filter3
    else:
        return filtered_image


def get_holomorfic(x, lower_freq=0.1, upper_freq=2.7, debug=False, mask=None):
    # How to se lower and upper freq... no idea
    # THere is a method by varying the frequencies and checking the
    # entropy between the corrected image and bias image
    x_scaled = harray.scale_minmax(x)
    x_256 = (255 * x_scaled).astype(np.uint8)

    # Create a mask
    if mask is None:
        mask_array = harray.get_otsu_mask(x)
    else:
        mask_array = mask

    # 1. Supress background..
    I_ROI = x_256 * mask_array

    # 2. Do a log transform
    I_log = np.log(I_ROI)
    I_log[np.isnan(I_log)] = 0
    I_log[np.isinf(I_log)] = 0

    # 3. Apply a filter in freq domain
    I_filt = butterworthbpf(I_log, d0=lower_freq, d1=upper_freq, n=2, debug=debug)
    if debug:
        I_mag = np.abs(I_filt[0])
        I_mag_filter = I_mag[1]
    else:
        I_mag = np.abs(I_filt)

    ROI_filt = butterworthbpf(mask_array, d0=lower_freq, d1=upper_freq, n=2, debug=debug)
    if debug:
        ROI_mag = np.abs(ROI_filt[0])
        I_ROI_filter = ROI_mag[1]
    else:
        ROI_mag = np.abs(ROI_filt)

    # 4. Divide the filtered image by the filtered ROI
    log_I_bias = I_mag / ROI_mag
    # 5. Subtract original image from bias field in the log-domain
    I_restored = np.exp(I_log - log_I_bias) * mask_array
    # 6. Retrieve the bias field (if needed)
    I_bias = np.exp(log_I_bias * mask_array)

    if debug:
        I_bias_masked = I_bias[mask_array == 1]
        bias_dens, _ = np.histogram(I_bias_masked, density=True)
        bias_entropy = scipy.stats.entropy(bias_dens)
        I_restore_masked = I_restored[mask_array == 1]
        restored_dens, _ = np.histogram(I_restore_masked, density=True)
        restored_entropy = scipy.stats.entropy(restored_dens)

        return I_restored, I_bias, restored_entropy, bias_entropy, I_mag_filter, I_ROI_filter

    else:
        return I_restored


if __name__ == "__main__":
    # Here we can test out this procedure....
    # Using measured data (read cpx)
    # Using measured data (read cpx)
    measured_path = '/home/bugger/Documents/data/7T/test_for_inhomog/prostate_7T'
    nibabel_dir = '/home/bugger/Documents/data/acdc_challenge_dataset/training/patient001/patient001_frame12.nii.gz'

    sel_slice_nibabel = 2
    nibabel_summed = nibabel.load(nibabel_dir).get_fdata().T[sel_slice_nibabel]
    nibabel_summed = harray.scale_minmax(nibabel_summed)
    hplotf.plot_3d_list(nibabel_summed)
    A_abs = nibabel_summed
    A_abs.max()
    res = get_holomorfic(A_abs)
    hplotc.ListPlot([A_abs, res])
    import helper.metric as hmetric
    hmetric.get_metrics_input(res, A_abs)
