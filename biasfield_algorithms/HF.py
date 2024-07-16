"""

HF - Homomorphic Filtering

HF is use as a classical bias
correction technique conducted on log-transformed image
intensities (P. A. Narayana & Borthakur, 1995; Johnston et
al., 1996; Velthuizen et al., 1998; Sreenivasan, Havlicek, &
Deshpande, 2015; Yang, Gach, Li, & Mutic, 2016). This
method able to simultaneously increase contrast and reduce
bias field. It extracts the bias field by low-pass filtering (LPF)
of the input image, then the corrected image can be obtained
by subtracting the bias field from the input image in the log-
domain, as expressed below:
log I(x, y) = log S (x, y) − LPF(log S (x, y)) + C N
(4)
where the the LPF(·) is the function of a low-pass filter, and
the C N is a normalization constant that used to keep the mean
or maximum intensity of the corrected image (Lewis & Fox,
2004).

I also added a method of my own.. that uses a different smoothing technique found online

"""

# Load 3T data
import nibabel
import helper.plot_class as hplotc
import SimpleITK as sitk
import helper.array_transf as harray
import scipy.signal
import numpy as np
import scipy.signal
from smoothing.L0_gradient_minimization import l0_gradient_minimization_2d


def get_hf(x, n_kernel=None, **kwargs):
    x = harray.scale_minmax(x)
    if n_kernel is None:
        n_kernel = int(x.shape[0] * 0.1)

    log_x = np.log(x)
    log_x[np.isnan(log_x)] = 0
    log_x[np.isinf(log_x)] = 0

    lpf_log_x = scipy.signal.convolve2d(log_x, np.ones((n_kernel, n_kernel)) / n_kernel ** 2, 'same')
    log_I = log_x - lpf_log_x
    I_recovered = np.exp(log_I)

    return I_recovered


def get_hf_l0(x, lmd=0.015, debug=False, mask=None):
    x = harray.scale_minmax(x)
    mask_array = harray.get_otsu_mask(x)
    lpf_x = l0_gradient_minimization_2d(x, lmd=lmd, beta_max=1e05, beta_rate=2.0)
    I_recovered = x / lpf_x * mask_array
    if debug:
        return I_recovered, lpf_x
    else:
        return I_recovered


if __name__ == "__main__":
    import os
    # Here we can test out this procedure....
    # Using measured data (read cpx)
    # Using measured data (read cpx)
    measured_path = '/home/bugger/Documents/data/7T/test_for_inhomog/prostate_7T'
    res = [os.path.join(measured_path, x) for x in os.listdir(measured_path)]
    hplotc.ListPlot(np.load(res[4]).sum(axis=0), augm='np.abs')
    
    nibabel_dir = '/home/bugger/Documents/data/acdc_challenge_dataset/training/patient001/patient001_frame12.nii.gz'

    sel_slice_nibabel = 2
    nibabel_summed = nibabel.load(nibabel_dir).get_fdata().T[sel_slice_nibabel]
    nibabel_summed = harray.scale_minmax(nibabel_summed)
    hplotc.ListPlot(nibabel_summed)
    A_abs = nibabel_summed
    A_abs.max()
    A_abs = harray.scale_minmax(A_abs)
    res = get_hf(A_abs)
    hplotc.ListPlot([A_abs, res])

    res = get_hf_l0(nibabel_summed, debug=True)
    zz = harray.scale_minpercentile_both(res[0], 99)
    harray.get_minmeanmediammax(zz)
    import matplotlib.pyplot as plt
    hplotc.ListPlot([A_abs, zz], vmin=(0,1))
