"""
First create a single cardiac SA mask..

Check the registration using a self created cardiac mask on the B1 data...
and link that with the mask on the MM1 data

This was not possible.. no rotation was there... or anything like that
"""

import nibabel
import os
import numpy as np
import helper.plot_class as hplotc
import helper.array_transf as harray
import skimage.transform

ddata_rho = '/media/bugger/MyBook/data/simulated/cardiac/bart/sa/rho'
ddata_mask = '/media/bugger/MyBook/data/simulated/cardiac/bart/sa/segm_mask'
rho_file_list = os.listdir(ddata_rho)
sel_rho_file = rho_file_list[0]

rho_array = np.load(os.path.join(ddata_rho, sel_rho_file))
rho_array = skimage.transform.rescale(rho_array, scale=3)
mask_array = np.load(os.path.join(ddata_mask, sel_rho_file))
mask_array = skimage.transform.rescale(mask_array, scale=3)

# Load the MMS mask
ddata = '/home/bugger/Documents/data/mm1/vendor_B/img'
ddata_label = '/home/bugger/Documents/data/mm1/vendor_B/label'
sel_file = os.listdir(ddata)[0]
ddata_file = os.path.join(ddata, sel_file)
ddata_file_label = os.path.join(ddata_label, sel_file)

loaded_array = nibabel.load(ddata_file).get_fdata()
loaded_array = np.moveaxis(loaded_array, -1, 0)
loaded_array_label = nibabel.load(ddata_file_label).get_fdata()
loaded_array_label = np.moveaxis(loaded_array_label, -1, 0)
nonzero_id = np.argwhere(np.sum(loaded_array_label, axis=(-2, -1)))
sel_label_mask = (loaded_array_label[nonzero_id[0]][0] > 0).astype(int)

import SimpleITK as sitk
fixed_image = sitk.GetImageFromArray(sel_label_mask.astype(int))
moving_image = sitk.GetImageFromArray(mask_array.astype(int))

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(fixed_image)
elastixImageFilter.SetMovingImage(moving_image)

parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.GetDefaultParameterMap("rigid"))
parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))

parameterMapVector.append(sitk.GetDefaultParameterMap("rigid"))
elastixImageFilter.SetParameterMap(parameterMapVector)

result_array = elastixImageFilter.Execute()
result_array = sitk.GetArrayFromImage(result_array)

hplotc.ListPlot([result_array, sel_label_mask])


temp_image = sitk.GetImageFromArray(rho_array)
transform_map = elastixImageFilter.GetTransformParameterMap()
validate_image = sitk.Transformix(temp_image, transform_map)
validate_array = sitk.GetArrayFromImage(validate_image)
hplotc.ListPlot(validate_array)


