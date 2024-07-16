import numpy as np
# Deze moet je blijven importeren
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.collections as mcol
import itertools
import helper.misc as hmisc
import helper.array_transf as harray
import scipy.optimize

from helper.plot_class import ListPlot, ImageIntensityEqualizer


if __name__ == "__main__":
    import skimage.data
    import torchio

    A = skimage.data.astronaut()[:, :, 0]
    A = harray.scale_minmax(A)
    corrupted_images = []
    list_of_biasfield_order = [0, 1, 2, 3, 4]
    for i_order in list_of_biasfield_order:
        gen_biasf = torchio.transforms.RandomBiasField(coefficients=1, order=i_order)
        A_transf = gen_biasf(A[None, :, :, None])[0, :, :, 0]
        A_transf = harray.scale_minmax(A_transf)
        corrupted_images.append(A_transf)

    img_equ = ImageIntensityEqualizer(reference_image=A, patch_width=100, image_list=corrupted_images,
                                      dynamic_thresholding=False, distance_measure='ssim')

    A_smooth_unity_equalized = img_equ.correct_image_list()
    """
    Below we will demonstrate the result of the equalization
    """
    subtitle_list = ['Original'] + [f'Order {i}' for i in list_of_biasfield_order]
    subtitle_list = [[x] for x in subtitle_list]
    plot_obj = ListPlot([A] + corrupted_images, cbar=True, col_row=(len(corrupted_images) + 1, 1), ax_off=True, subtitle=subtitle_list)
    img_equ.plot_crop_coords(plot_obj.ax_list[0])
    plot_obj = ListPlot([A] + A_smooth_unity_equalized, cbar=True, col_row=(len(corrupted_images) + 1, 1), ax_off=True, subtitle=subtitle_list)

    """
    Here we are going to test each component of the equalization
    """
    # First get the mean scaling values
    mean_scaling_list = img_equ.get_mean_scaling_from_patches()
    mean_corrected_image_list = img_equ.apply_mean_scaling(mean_scaling_list)
    corrected_image_list = img_equ.apply_vmax_ref(mean_corrected_image_list)
    ListPlot(img_equ.image_list, cbar=True)
    ListPlot(mean_corrected_image_list, cbar=True)
    ListPlot(corrected_image_list, cbar=True)
    ListPlot(A_smooth_unity_equalized, cbar=True)

    patch_corrected_images = img_equ.get_patches_image_list(image_list=corrected_image_list)
    average_ref_patch_intensity = np.mean(img_equ.patches_ref)
    std_ref_patch_intensity = np.std(img_equ.patches_ref)
    average_cor_patch_intensity = np.array([np.mean(x) for x in patch_corrected_images])
    print("Fraction between average ref patch and cor patch intensity AFTER mean correction\n")
    for ii, i_fraction in enumerate(average_ref_patch_intensity / average_cor_patch_intensity):
        print(f'\tImage {ii}: {i_fraction}')
    ListPlot(corrected_image_list, vmin=(average_ref_patch_intensity - std_ref_patch_intensity, average_ref_patch_intensity + std_ref_patch_intensity))
    """
    Below we demonstrate the effect of the equalization in an average setting
    """

    uncorrected_patches = img_equ.get_patches_image_list(image_list=corrupted_images)
    corrected_patches = img_equ.get_patches_image_list(image_list=A_smooth_unity_equalized)
    difference_uncor = np.round(np.linalg.norm(img_equ.patches_ref - np.array(uncorrected_patches), axis=(-1, -2)), 2)
    difference_cor = np.round(np.linalg.norm(img_equ.patches_ref - np.array(corrected_patches), axis=(-1, -2)), 2)
    print('Average difference when uncorrected ', np.mean(difference_uncor).round(2))
    print('Average difference when corrected ', np.mean(difference_cor).round(2))
    print('Ratio ', np.mean(difference_cor).round(2) / np.mean(difference_uncor).round(2))

    """
    Quickly compare with histogram equilization
    """
    import cv2
    import matplotlib.pyplot as plt
    from skimage.util import img_as_ubyte

    A_hist_equalized = cv2.equalizeHist(img_as_ubyte(A))
    corrupted_images_hist_equalized = [cv2.equalizeHist(img_as_ubyte(x)) for x in corrupted_images]
    # Visualize inspect the correction
    ListPlot([A_hist_equalized] + corrupted_images_hist_equalized, cbar=True)

    reference_path = img_equ.get_patches_image_list(image_list=[A_hist_equalized])[0]
    uncorrected_patches = img_equ.get_patches_image_list(image_list=corrupted_images)
    corrected_patches = img_equ.get_patches_image_list(image_list=corrupted_images_hist_equalized)
    difference_uncor = np.round(np.linalg.norm(reference_path - np.array(uncorrected_patches), axis=(-1, -2)), 2)
    difference_cor = np.round(np.linalg.norm(reference_path - np.array(corrected_patches), axis=(-1, -2)), 2)
    print('Average difference when uncorrected ', np.mean(difference_uncor).round(2))
    print('Average difference when corrected ', np.mean(difference_cor).round(2))
    print('Ratio ', np.mean(difference_cor).round(2) / np.mean(difference_uncor).round(2))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    A_CLAHE = clahe.apply(img_as_ubyte(A).astype(np.uint8))
    corrupted_images_CLAHE = [cv2.equalizeHist(img_as_ubyte(x).astype(np.uint8)) for x in corrupted_images]
    ListPlot([A_CLAHE] + corrupted_images_CLAHE, cbar=True)

    reference_path = img_equ.get_patches_image_list(image_list=[A_CLAHE])[0]
    uncorrected_patches = img_equ.get_patches_image_list(image_list=corrupted_images)
    corrected_patches = img_equ.get_patches_image_list(image_list=corrupted_images_CLAHE)
    difference_uncor = np.round(np.linalg.norm(reference_path - np.array(uncorrected_patches), axis=(-1, -2)), 2)
    difference_cor = np.round(np.linalg.norm(reference_path - np.array(corrected_patches), axis=(-1, -2)), 2)
    print('Average difference when uncorrected ', np.mean(difference_uncor).round(2))
    print('Average difference when corrected ', np.mean(difference_cor).round(2))
    print('Ratio ', np.mean(difference_cor).round(2) / np.mean(difference_uncor).round(2))