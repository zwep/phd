import helper.array_transf as harray
import skimage.transform
import SimpleITK as sitk
import numpy as np
import helper.plot_class as hplotc
import h5py
import os


def print_parameter_map(x, space_dist=40):
    for k, v in x.items():
        print(k, (space_dist - len(k)) * ' ', v)

"""
Lets try this...
"""

ddata_base = '/home/bugger/Documents/data/3T/prostate/prostate_weighting/test'
ddata_1p5T = os.path.join(ddata_base, 'input')
ddata_mask = os.path.join(ddata_base, 'mask')
ddata_3T = os.path.join(ddata_base, 'target')
ddata_3T_cor = os.path.join(ddata_base, 'target_corrected')

ddata_3T_cor_regular = os.path.join(ddata_base, 'target_regular_corrected')
ddata_3T_cor_gan = os.path.join(ddata_base, 'target_gan_corrected')

ddata_segm_mask = os.path.join(ddata_base, 'segmentation', '7_MR.npy')

# This is from slice...
sel_slice_3T = 40
sel_slice_1p5T = 47

with h5py.File(os.path.join(ddata_3T, '7_MR.h5'), 'r') as f:
    input_array = np.array(f['data'][sel_slice_3T])

with h5py.File(os.path.join(ddata_3T_cor_gan, '7_MR.h5'), 'r') as f:
    gan_array = np.array(f['data'][sel_slice_3T])

with h5py.File(os.path.join(ddata_3T_cor_regular, '7_MR.h5'), 'r') as f:
    regular_array = np.array(f['data'][sel_slice_3T])

with h5py.File(os.path.join(ddata_3T_cor, '7_MR.h5'), 'r') as f:
    corrected_array = np.array(f['data'][sel_slice_3T])

with h5py.File(os.path.join(ddata_1p5T, '7_MR.h5'), 'r') as f:
    target_array = np.array(f['data'][sel_slice_1p5T])


with h5py.File(os.path.join(ddata_mask, '7_MR_input.h5'), 'r') as f:
    mask_array_1p5T = np.array(f['data'][sel_slice_1p5T])


with h5py.File(os.path.join(ddata_mask, '7_MR_target.h5'), 'r') as f:
    mask_array_3T = np.array(f['data'][sel_slice_3T])

"""
Make sure stuff is aligned
"""

target_array = harray.scale_minmax(target_array)

moving_array_list = [input_array, corrected_array, regular_array, gan_array]
moving_array_list = [harray.scale_minmax(x) for x in moving_array_list]

# # # Toy a bit with metrics in the parameter map
fixed_array, moving_array, fixed_mask, moving_mask = harray.rigid_align_images(target_array, moving_array_list[0])
hplotc.ListPlot([fixed_array, moving_array], cbar=True)

dpoint_file = "/home/bugger/Documents/data/3T/prostate/prostate_weighting/test/fixed_point_set.pts"
# Create the fixed image and fixed mask image..
fixed_image = sitk.GetImageFromArray(fixed_array)
fixed_mask_image = sitk.GetImageFromArray(fixed_mask.astype(int))
fixed_mask_image = sitk.Cast(fixed_mask_image, sitk.sitkUInt8)

# Create the moving image and moving mask image..
moving_image = sitk.GetImageFromArray(moving_array)
moving_mask_image = sitk.GetImageFromArray(moving_mask.astype(int))
moving_mask_image = sitk.Cast(moving_mask_image, sitk.sitkUInt8)

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetOutputDirectory('/home/bugger/Documents/data/output_elastix')
elastixImageFilter.LogToConsoleOn()
elastixImageFilter.LogToFileOn()
# Set images
elastixImageFilter.SetFixedImage(fixed_image)
elastixImageFilter.SetFixedMask(fixed_mask_image)
elastixImageFilter.SetMovingImage(moving_image)
elastixImageFilter.SetMovingMask(moving_mask_image)

# COnfigure an affine map..
affine_map = sitk.GetDefaultParameterMap("affine")
print_parameter_map(affine_map)

# Configure Bspline map....
bspline_map = sitk.GetDefaultParameterMap("bspline")
print_parameter_map(bspline_map)
del bspline_map['Metric0Weight']
del bspline_map['Metric1Weight']
bspline_map["Metric"] = ("CorrespondingPointsEuclideanDistanceMetric", )
# elastixImageFilter.SetParameterMap(affine_map)
elastixImageFilter.SetParameterMap(bspline_map)
# elastixImageFilter.SetFixedPointSetFileName(dpoint_file)

elastixImageFilter.PrintParameterMap()

elastixImageFilter.Execute()
res = elastixImageFilter.GetResultImage()
res_array = sitk.GetArrayFromImage(res)
hplotc.ListPlot([res_array, moving_array, fixed_array])