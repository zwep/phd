# encoding: utf-8

# https://simpleelastix.readthedocs.io/GettingStarted.html

import numpy as np
import os
import importlib
import matplotlib.pyplot as plt

import SimpleITK as sitk
import nrrd
import scipy.signal
import scipy.ndimage

"""
Simple file to explore the posibilities of registration
"""

N = 25
A = np.zeros((N, N, N))
min_ind = N//2 - N//4
max_ind = N//2 + N//4
A[min_ind: max_ind, min_ind: max_ind, min_ind: max_ind] = 1
# Create a rotatated image
A_rot = scipy.ndimage.rotate(A, angle=30, axes=(-2, -1), reshape=False)

# Create an affine transformation
A_affine = np.zeros(A_rot.shape)
# This 3 and 21 are found by inspection...
min_x = min_y = min_z = 3
max_x = max_y = max_z = 21
affine_x, affine_y, affine_z = np.random.randint(0, 5, size=3)
min_x_shifted = min_x + affine_x
max_x_shifted = max_x + affine_x
min_y_shifted = min_y + affine_y
max_y_shifted = max_y + affine_y
min_z_shifted = min_z + affine_z
max_z_shifted = max_z + affine_z
cropped_x = A_rot[min_z:max_z, min_y:max_y, min_x:max_x]
A_affine[min_z_shifted: max_z_shifted, min_y_shifted: max_y_shifted, min_x_shifted: max_x_shifted] = cropped_x

fixed_image = sitk.GetImageFromArray(A)
moving_image = sitk.GetImageFromArray(A_affine)

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(fixed_image)
elastixImageFilter.SetMovingImage(moving_image)

parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
parameterMapVector.append(sitk.GetDefaultParameterMap("rigid"))
elastixImageFilter.SetParameterMap(parameterMapVector)

result_array = elastixImageFilter.Execute()
result_array = sitk.GetArrayFromImage(result_array)

transform_map = elastixImageFilter.GetTransformParameterMap()
validate_image = sitk.Transformix(moving_image, transform_map)
validate_array = sitk.GetArrayFromImage(validate_image)
