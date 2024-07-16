"""
Some parts are really really weird.

I am using here anything BUT my implemented classes, just to be sure.

"""

# Check the performance of the registration....
import helper.plot_class as hplotc
import helper.misc as hmisc
import helper.array_transf as harray
import numpy as np
import h5py
import scipy.io
import skimage.transform as sktransf
import data_prep.registration.Registration as Registration

with h5py.File(ddata, 'r') as f:
    res = np.array(f['data'])

def read_h5(file, key=None):
    with h5py.File(file, 'r') as f:
        res = np.array(f[key])
    return res


b1_file = '/home/bugger/Documents/data/test_clinic_registration/flavio_data/M01.mat'
mask_file = '/home/bugger/Documents/data/test_clinic_registration/mri_data/mask_h5/1_MR/MRL/20210112_0002.h5'
rho_file = '/home/bugger/Documents/data/test_clinic_registration/mri_data/prostate_h5/1_MR/MRL/20210112_0002.h5'

mask_array = read_h5(mask_file, 'data')
rho_array = read_h5(rho_file, 'data')

b1_obj = scipy.io.loadmat(b1_file)
b1_mask = b1_obj['Model']['Mask'][0][0]
b1_plus = np.moveaxis(b1_obj['Model']['B1plus'][0][0], -1, 0)
b1_minus = np.moveaxis(b1_obj['Model']['B1minus'][0][0], -1, 0)
hplotc.ListPlot([b1_plus], augm='np.abs', ax_off=True)

sel_slice = 35
mask_patient_array = mask_array[sel_slice]
rho_patient_array = rho_array[sel_slice]

new_shape = rho_array.shape[-2:]
hplotc.ListPlot([mask_array[sel_slice], rho_array[sel_slice]])

b1_plus_resize = np.array([harray.resize_complex_array(x, new_shape=new_shape, preserve_range=True) for x in b1_plus])
b1_minus_resize = np.array([harray.resize_complex_array(x, new_shape=new_shape, preserve_range=True) for x in b1_minus])
b1_mask_resize = sktransf.resize(b1_mask, new_shape, preserve_range=True)

hplotc.ListPlot(b1_plus_resize[None], augm='np.abs', start_square_level=2, ax_off=True)

from_image = b1_mask_resize.astype(int)
to_image = mask_patient_array.astype(int)

import SimpleITK as sitk
# Get the registation parameters
registration_parameter = sitk.VectorOfParameterMap()
rigid_map = sitk.GetDefaultParameterMap("rigid")
affine_map = sitk.GetDefaultParameterMap("affine")
bspline_map = sitk.GetDefaultParameterMap("bspline")
registration_parameter.append(bspline_map)

# Create a registration object
elastix_obj = sitk.ElastixImageFilter()

# Convert to Image object
moving_image = sitk.GetImageFromArray(from_image)
fixed_image = sitk.GetImageFromArray(to_image)
# Set the images...
elastix_obj.SetMovingImage(moving_image)
elastix_obj.SetFixedImage(fixed_image)
# Start the registration
elastix_obj.SetParameterMap(registration_parameter)
elastix_obj.Execute()
# Get the transformation map
transform_mapping = elastix_obj.GetTransformParameterMap()


# Now check how well we did..
x_image = sitk.GetImageFromArray(from_image)
result_image = sitk.Transformix(x_image, transform_mapping)
result_array = sitk.GetArrayFromImage(result_image)

hplotc.ListPlot([from_image, result_array, to_image])

# Now check the performance on the original stuff..
def registrate_array(x):
    res = []
    for i_image in x:
        x_image = sitk.GetImageFromArray(i_image)
        result_image = sitk.Transformix(x_image, transform_mapping)
        result_array = sitk.GetArrayFromImage(result_image)
        res.append(result_array)
    return np.array(res)

b1_plus_reg_real = registrate_array(b1_plus_resize.real)
b1_plus_reg_imag = registrate_array(b1_plus_resize.imag)
b1_plus_reg_cpx = b1_plus_reg_real + 1j * b1_plus_reg_imag
hplotc.ListPlot([b1_plus_reg_cpx.sum(axis=0)], augm='np.abs')

"""
Apparantly things went.. well?

Now try this with the original classes...
"""

registration_obj = Registration.Registration(A=np.abs(b1_plus_resize).sum(axis=0), B=rho_patient_array,
                                             A_mask=b1_mask_resize.astype(int), B_mask=mask_patient_array.astype(int),
                                             registration_options='')

_ = registration_obj.register_mask()
result_reg_mask = registration_obj.apply_registration(b1_mask_resize)
temp_dice_score = hmisc.dice_metric(result_reg_mask, mask_patient_array)
print('Current dice score ', temp_dice_score)
registration_obj.display_content()
registration_obj.display_mask_validation()

b1p_registered = np.array([registration_obj.apply_registration_cpx(x) for x in b1_plus_resize])
hplotc.ListPlot(b1p_registered.sum(axis=0), augm='np.abs')

"""
So that one also worked..

This one did not work in the beginning... but... after some type conversion it looks OK now.
"""

import data_prep.registration.RegistrationProcess as RegistrationProcess
import importlib
importlib.reload(RegistrationProcess)
regproc_obj = RegistrationProcess.RegistrationProcess(patient_files=[rho_file],
                                                      patient_mask_files=[mask_file],
                                                      b1_file=b1_file,
                                                      dest_path='/home/bugger',
                                                      data_type='test',
                                                      display=False,
                                                      registration_options='',
                                                      n_cores=2)

res = regproc_obj.run_slice(0, 35)
regproc_obj.registration_obj.display_content()
regproc_obj.get_current_status()

b1p_registered, b1m_registered, patient_slice, patient_mask_slice = res
hplotc.ListPlot(b1p_registered.sum(axis=0), augm='np.abs')
hplotc.ListPlot(b1p_registered, augm='np.abs')

