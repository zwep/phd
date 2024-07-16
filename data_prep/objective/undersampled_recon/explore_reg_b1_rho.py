# encoding: utf-8

# https://simpleelastix.readthedocs.io/GettingStarted.html

import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc

import SimpleITK as sitk
import nrrd
import data_generator.Rx2Tx as gen_rx2tx
import scipy.signal
import scipy.ndimage

"""
Simple file to explore the posibilities of registration
"""

# Load flavio data
importlib.reload(gen_rx2tx)
dir_data_flavio = '/home/bugger/Documents/data/simulation/flavio_npy'
dg_gen_rx2tx_flavio = gen_rx2tx.DataSetSurvey2B1_flavio(input_shape=(2, 512, 256),
                                                        ddata=dir_data_flavio,
                                                        masked=True)

a, b, flavio_mask = dg_gen_rx2tx_flavio.__getitem__(0)


# Load nrrd data
data_dir = '/home/bugger/Documents/data/prostatemriimagedatabase'
file_dir_list = sorted([x for x in os.listdir(data_dir) if x.endswith('.nrrd')])

i_file = file_dir_list[0]
file_path = os.path.join(data_dir, i_file)

temp_data, temp_header = nrrd.read(file_path)
temp_data = np.moveaxis(temp_data, -1, 0)[np.newaxis]

# Loop over images...
input_array = temp_data[0, 0]
input_array = np.flipud(np.rot90(input_array))
filled_mask = harray.get_smoothed_mask(input_array)


flavio_mask_image = sitk.GetImageFromArray(flavio_mask)
prostate_mask_image = sitk.GetImageFromArray(filled_mask)

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(prostate_mask_image)
elastixImageFilter.SetMovingImage(flavio_mask_image)

parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
elastixImageFilter.SetParameterMap(parameterMapVector)

result_array = elastixImageFilter.Execute()
result_array = sitk.GetArrayFromImage(result_array)

transform_map = elastixImageFilter.GetTransformParameterMap()
validate_image = sitk.Transformix(flavio_mask_image, transform_map)
validate_array = sitk.GetArrayFromImage(validate_image)
hplotf.plot_3d_list([result_array, validate_array, result_array - validate_array])

transform_map = elastixImageFilter.GetTransformParameterMap()
# Store transform map...
# sitk.WriteParameterFile(transform_map[0], '/home/bugger/derp.txt')
# Flavio id_to_prostateX...

# Apply transformation
a_sel = a[0]
a_image = sitk.GetImageFromArray(a_sel)
resultLabel = sitk.Transformix(a_image, transform_map)
res_array_3 = sitk.GetArrayFromImage(resultLabel)

a_sel = a[1]
a_image = sitk.GetImageFromArray(a_sel)
resultLabel = sitk.Transformix(a_image, transform_map)
res_array_4 = sitk.GetArrayFromImage(resultLabel)

z_real = res_array_3 * filled_mask.astype(np.float32)
# Maybe this should also be corrected OUTSIDE the mask..?
z_real[np.isclose(z_real, 0)] = 0

z_imag = res_array_4 * filled_mask.astype(np.float32)
# Maybe this should also be corrected OUTSIDE the mask..?
z_imag[np.isclose(z_imag, 0)] = 0
z = z_real + 1j * z_imag

hplotf.plot_3d_list([a_sel, res_array_3, res_array_4], cbar=True, vmin=(-0.5, 0.5))
