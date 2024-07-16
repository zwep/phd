import numpy as np
import helper.misc as hmisc
import helper.array_transf as harray
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte, img_as_uint

from small_project.GLCM.check_glcm_feature_breeuwer import StabilityGLCM


def convert_img(img, n=8):
    img_int = img * (2 ** n - 1)
    img_int_ceil = np.ceil(img_int)
    img_int_floor = np.floor(img_int)
    if n == 8:
        img_new = img_as_ubyte(img)
    elif n == 16:
        img_new = img_as_uint(img)
    else:
        img_new = None
    return img_int, img_int_ceil, img_int_floor, img_new


def get_GLCM_of_img(img, convert_patch=False, add_worst_noise=False):
    stab_obj = StabilityGLCM(img)

    # Get patches
    patch_shape = tuple(np.array(img.shape) // 10)
    stride = min(patch_shape)
    temp_patches = harray.get_patches(img, patch_shape=patch_shape, stride=stride)
    print(f'Patch shape {patch_shape}')
    print(f'Patch stride {stride}')

    if add_worst_noise:
        # TODO: add this
        for sel_patch in temp_patches:
            sel_patch = harray.scale_minmax(sel_patch)
            residual = sel_patch % 1
            sel_patch[residual >= 0.5] = np.floor(sel_patch[residual >= 0.5])
            sel_patch[residual < 0.5] = np.ceil(sel_patch[residual < 0.5])

    if convert_patch:
        temp_patches = [img_as_ubyte(harray.scale_minmax(x)) for x in temp_patches]

    metric_list = ['homogeneity', 'energy']
    metric_dict = {}
    for sel_index, sel_patch in enumerate(temp_patches):
        sel_glcm_obj = stab_obj.get_GLCM_obj(sel_patch)
        for sel_metric_name in metric_list:
            metric_dict.setdefault(sel_metric_name, [])
            main_glcm_metric = stab_obj._get_GLCM_metric(sel_glcm_obj, sel_metric_name)[0][0]
            metric_dict[sel_metric_name].append(main_glcm_metric)
    return metric_dict


"""

Experiments I want to perform

o float image -> uint8
    calculate GLCM metrics over patch

o float image -> patches
    patches -> uint8
        calculte GLCM metrics over patch

o Same experiments but with added 'worst' noise 

o float images -> custom range 
    calculate GLCM metrics over patch
        -- vary range and compare values

"""

ddata = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/t2w/v9_18012021_0939588_10_3_t2wV4.npy'

A = hmisc.load_array(ddata)
A_sos = np.sqrt(np.sum(np.abs(A) ** 2, axis=0))
A_sos = harray.scale_minmax(A_sos)

img_int_8_approx, img_int_8_ceil, img_int_8_floor, img_int_8 = convert_img(A_sos, 8)
img_int_16_approx, img_int_16_ceil, img_int_16_floor, img_int_16 = convert_img(A_sos, 16)

glcm_result_global_convert = get_GLCM_of_img(img_int_8, convert_patch=False)
glcm_result_local_convert = get_GLCM_of_img(A_sos, convert_patch=True)


fig, ax = plt.subplots()
metric_key = 'energy'
ax.hist(glcm_result_global_convert[metric_key], label='global', color='r')
ax.vlines(np.mean(glcm_result_global_convert[metric_key]), ymin=0, ymax=100, color='k')
ax.hist(glcm_result_local_convert[metric_key], label='local', color='b')
ax.vlines(np.mean(glcm_result_local_convert[metric_key]), ymin=0, ymax=100, color='k')
plt.legend()

#
import os
import helper.array_transf as harray
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import data
import skimage.util
import matplotlib.pyplot as plt
import harreveltools
import skimage.data
import skimage.exposure


def get_quantization(img, glcm_metric='energy', n_start=256, n_stop=1024):
    metric_value = []
    for ii in range(n_start, n_stop):
        A_custom = (img * ii).astype(int)
        P = graycomatrix(
                    A_custom,
                    distances=[1],
                    angles=[0],
                    levels=ii+1,
                    symmetric=True,
                    normed=True,
                )
        z = graycoprops(P, glcm_metric)[0][0]
        metric_value.append(z)
    return metric_value

# Get an image
image = skimage.data.astronaut()[..., 0].astype(float)
# Add noise and make it a float
image += np.random.random(image.shape)
# Rescale to 0..1 for later conversion
image = skimage.exposure.rescale_intensity(image)
metric_value_quantization = get_quantization(image)

fig, ax = plt.subplots()
ax.plot(range(256, 1024), metric_value_quantization, 'r')
plt.title('GLCM metric over different quantization levels')
ax.set_xlabel('number of bins')
ax.set_ylabel('GLCM metric value')


import numpy as np


def eqProbQuant(image, levels=32):
    # Sort the pixels by value
    sorted_image = np.sort(image.ravel())
    # Get the pixel count
    pixel_count = sorted_image.shape[0] - 1
    # Get the number of pixels per bin
    samples_per_bin = int(pixel_count / levels)
    print(samples_per_bin)
    # Get locations where the bin would change
    edge_samples = np.arange(levels + 1) * samples_per_bin
    # Get the values at those locations (bin edges)
    bin_edges = sorted_image[edge_samples]
    # Use the values to apply quantization
    quantized = np.digitize(image, bin_edges)

    return quantized

import helper.plot_class as hplotc

x = temp_patches[sel_index]
xnorm = harray.scale_minmax(temp_patches[sel_index])

quantized_32 = eqProbQuant(xnorm, 31)
quantized_64 = eqProbQuant(xnorm, 63)
quantized_256 = eqProbQuant(xnorm, 255)

hplotc.ListPlot([quantized_32, quantized_64, quantized_256, img_as_ubyte(xnorm), img_as_ubyte(x)])

P = graycomatrix(
    quantized_256,
    distances=[1],
    angles=[0],
    levels=256 + 1,
    symmetric=True,
    normed=True,
)
graycoprops(P, 'energy')[0][0]