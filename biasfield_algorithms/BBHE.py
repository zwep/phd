"""
Fancy Histogram Equilazation

Arriaga-Garcia, Edgar F., Raul E. Sanchez-Yanez, and M. G. Garcia-Hernandez.

"Image enhancement using bi-histogram equalization with adaptive sigmoid functions."
2014 International Conference on Electronics, Communications and Computers (CONIELECOMP). IEEE, 2014.
"""

import numpy as np
import helper.array_transf as harray
import os
import helper.plot_class as hplotc
import helper.plot_fun as hplotf
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
from skimage.util import img_as_ubyte, img_as_uint



def get_bbhe(img, max_int=256, **kwargs):
    """
    :param img: From the implementation point of view, it is still a grayscale image
         :return: return the modified image

     Usd https://www.programmersought.com/article/26045370638/
    """
    if max_int == 2 ** 8:
        img = img_as_ubyte(img)
    elif max_int == 2 ** 16:
        img = img_as_uint(img)
    else:
        print('Unknown int type..')

    xmin = np.min(img)
    xmax = np.max(img)
    img_result = np.zeros_like(img)
    # Average gray value
    xm = np.mean(img)

    sl = np.zeros((max_int, 1))     # The image is divided into two parts l and u
    su = np.zeros((max_int, 1))
    nl = 0
    nu = 0

    # Count the pixels in the image and classify them
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < xm:
                sl[img[i, j] - 1] += 1
                nl += 1
            else:
                su[img[i, j] - 1] += 1
                nu += 1
    pl = sl / nl
    pu = su / nu

    hist_cl = np.cumsum(pl)
    hist_cu = np.cumsum(pu)

    hist_cl = xmin + hist_cl * (xm - xmin)
    hist_cu = xm + 1 + hist_cu * (xmax - xm -1)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] <= xm:
                temp = img[i, j]
                img_result[i, j] = hist_cl[temp - 1]
            else:
                temp = img[i, j]
                img_result[i, j] = hist_cu[temp - 1]

    return img_result


def hsv_bbhe(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = get_bbhe(img_gray).astype(np.uint8)
    result = cv2.merge((h, s, img_gray))
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    return result


if __name__ == "__main__":
    # Check if we can implemetn BBHE

    # Load 3T data
    import nibabel
    import helper.plot_class as hplotc
    import SimpleITK as sitk
    import helper.array_transf as harray
    import scipy.signal
    import helper.plot_class as hplotc
    # Using measured data (read cpx)
    measured_path = '/home/bugger/Documents/data/7T/test_for_inhomog/prostate_7T'
    nibabel_dir = '/home/bugger/Documents/data/acdc_challenge_dataset/training/patient001/patient001_frame12.nii.gz'

    sel_slice_nibabel = 2
    nibabel_summed = nibabel.load(nibabel_dir).get_fdata().T[sel_slice_nibabel]
    nibabel_summed = harray.scale_minmax(nibabel_summed)
    hplotc.ListPlot(nibabel_summed)
    A_abs = nibabel_summed
    A_abs.max()
    A_256 = (A_abs * 255).astype(np.int)
    res = get_bbhe(A_256)
    hplotf.plot_3d_list([res, A_256])


    # Check if we can do some kind of histogram equalization....
    # Should check these values with real data...

    import scipy.optimize





    import skimage.data
    import helper.array_transf as harray
    import helper.plot_class as hplotc
    import helper.metric as hmetric
    import matplotlib.pyplot as plt
    import numpy as np
    A = skimage.data.astronaut()[:, :, 0]
    A_unity = harray.scale_minmax(A)
    A_smooth = np.round(harray.smooth_image(A, 64)).astype(A.dtype)
    A_smooth_unity = harray.scale_minmax(A_smooth)
    hplotc.ListPlot([A, A_smooth])
    A_cov = hmetric.coefficient_of_variation(A)
    A_smooth_cov = hmetric.coefficient_of_variation(A_smooth)
    print(f"Cov {round(A_cov,2)} Smooth cov {round(A_smooth_cov,2)}")
    fig, ax = plt.subplots()
    _ = ax.hist(A.ravel(), bins=range(255), label='normal', alpha=0.5)
    _ = ax.hist(A_smooth.ravel(), bins=range(255), label='smooth', alpha=0.5)
    ax.legend()
    print(hmetric.relative_coefficient_of_variation(A_unity, A_smooth_unity))
    print(np.linalg.norm(A_unity - A_smooth_unity))
    """# Drie methoden?
    # Image Equalizer met patch"""
    equil_obj = hplotc.ImageIntensityEqualizer(reference_image=A_unity, image_list=[A_smooth_unity])
    A_smooth_unity_equalized = equil_obj.correct_image_list()[0]
    fig, ax = plt.subplots()
    _ = ax.hist(A_unity.ravel(), bins=np.linspace(0, 1, 255), label='normal', alpha=0.5)
    _ = ax.hist(A_smooth_unity_equalized.ravel(), bins=np.linspace(0, 1, 255), label='smooth', alpha=0.5)
    ax.legend()
    hplotc.ListPlot([res, A_smooth, A])
    print(hmetric.relative_coefficient_of_variation(A_unity, A_smooth_unity_equalized))
    print(np.linalg.norm(A_unity - A_smooth_unity_equalized))
    mean_unity = np.mean(A_unity)
    mean_smooth_unity = np.mean(A_smooth_unity)
    A_smooth_unity_demeaned = mean_unity / mean_smooth_unity * A_smooth_unity
    fig, ax = plt.subplots()
    _ = ax.hist(A_unity.ravel(), bins=np.linspace(0, 1, 255), label='normal', alpha=0.5)
    _ = ax.hist(A_smooth_unity_demeaned.ravel(), bins=np.linspace(0, 1, 255), label='smooth', alpha=0.5)
    ax.legend()
    hplotc.ListPlot([A_smooth_unity_demeaned, A_unity])
    print(hmetric.relative_coefficient_of_variation(A_unity, A_smooth_unity_demeaned))
    print(np.linalg.norm(A_unity - A_smooth_unity_demeaned))
    """# Mean/Median scaling?"""
    mean_unity = np.mean(A_unity)
    mean_smooth_unity = np.mean(A_smooth_unity)
    A_smooth_unity_demeaned = mean_unity / mean_smooth_unity * A_smooth_unity
    median_unity = np.median(A_unity)
    median_smooth_unity = np.median(A_smooth_unity)
    A_smooth_unity_demedianed = median_unity / median_smooth_unity * A_smooth_unity
    A_smooth_unity_demedianed_demeaned = (A_smooth_unity_demedianed + A_smooth_unity_demeaned)/2
    fig, ax = plt.subplots()
    _ = ax.hist(A_unity.ravel(), bins=np.linspace(0, 1, 255), label='normal', alpha=0.5)
    _ = ax.hist(A_smooth_unity_demedianed_demeaned.ravel(), bins=np.linspace(0, 1, 255), label='smooth', alpha=0.5)
    ax.legend()
    print(hmetric.relative_coefficient_of_variation(A_unity, A_smooth_unity_demedianed_demeaned))
    print(np.linalg.norm(A_unity - A_smooth_unity_demedianed_demeaned))
    """# Linear line scaling?"""
    nx, ny = A_unity.shape
    x_pred_line = A_unity[:, nx // 2]
    x_target_line = A_smooth_unity[:, nx // 2]
    # Create a plot based on lines....
    minimize_obj = MinimizeL2(x_pred_line, x_target_line)
    result_min = minimize_obj.minimize_run()
    a, b = result_min
    A_smooth_unity_affine = A_smooth_unity * a + b
    A_smooth_unity_affine = harray.scale_minmax(A_smooth_unity_affine)
    hplotc.ListPlot([A_smooth_unity_affine- A_smooth_unity], ax_off=True, cbar=True)
    hplotc.ListPlot([A_smooth_unity_affine- A_smooth_unity, A_smooth_unity, A_unity])
    fig, ax = plt.subplots()
    _ = ax.hist(A_unity.ravel(), bins=np.linspace(0, 1, 255), label='normal', alpha=0.5)
    _ = ax.hist(A_smooth_unity_affine.ravel(), bins=np.linspace(0, 1, 255), label='smooth', alpha=0.5)
    ax.legend()
    print(hmetric.relative_coefficient_of_variation(A_unity, A_smooth_unity_affine))
    print(np.linalg.norm(A_unity - A_smooth_unity_affine))