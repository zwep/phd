import numpy as np
import os
import helper.array_transf as harray
import helper.plot_class as hplotc
import reconstruction.ReadCpx as read_cpx
import scipy.io

ddata = '/media/bugger/MyBook/data/7T_data/cardiac_radial/V9_16051/v9_09122020_1704205_7_2_transradialfastV4.mat'
ddest = '/home/bugger/Documents/data/7T/kspace_precision_test'

"""
Cpxdata
"""

# Example of float32/complex64 data
A = scipy.io.loadmat(ddata)['reconstructed_data']
A = np.squeeze(A)
A = np.sum(A, axis=2)
A = A[:, :, 4]

A_img = harray.transform_kspace_to_image_fftn(A)
A_cpx_128 = harray.transform_kspace_to_image_fftn(A.astype(np.complex128).astype(np.complex64))
A_cpx_64 = harray.transform_kspace_to_image_fftn(A.astype(np.complex64).astype(np.complex64))
hplotc.ListPlot([[A_img - A_cpx_128, A_img - A_cpx_64]], augm='np.abs')

B = harray.to_stacked(A, 'cartesian', stack_ax=-1)
B_img = harray.transform_kspace_to_image_fftn(harray.to_complex(B))
B_float_32 = harray.transform_kspace_to_image_fftn(harray.to_complex(B.astype(np.float32)).astype(np.complex64))
B_float_16 = harray.transform_kspace_to_image_fftn(harray.to_complex(B.astype(np.float16)).astype(np.complex64))

hplotc.ListPlot([[B_img - B_float_32, B_img - B_float_16]], augm='np.abs')

B_int_8 = harray.transform_kspace_to_image_fftn(harray.to_complex(B.astype(np.int8)).astype(np.complex64))
B_int_16 = harray.transform_kspace_to_image_fftn(harray.to_complex(B.astype(np.int16)).astype(np.complex64))
B_int_32 = harray.transform_kspace_to_image_fftn(harray.to_complex(B.astype(np.int32)).astype(np.complex64))
B_int_64 = harray.transform_kspace_to_image_fftn(harray.to_complex(B.astype(np.int64)).astype(np.complex64))

hplotc.ListPlot([[B_img - B_int_8, B_img - B_int_16, B_img - B_int_32, B_img - B_int_64]], augm='np.abs')

"""
Test Dtype of Flavio Data
"""

# Example of float64/complex128 data
ddata = '/home/bugger/Documents/data/test_clinic_registration/flavio_data/M01.mat'
A_img = scipy.io.loadmat(ddata)['Model']['B1plus'][0][0].sum(axis=-1)

A_cpx_128 = A_img.astype(np.complex128).astype(np.complex128)
A_cpx_64 = A_img.astype(np.complex64).astype(np.complex128)
hplotc.ListPlot([[A_img - A_cpx_128, A_cpx_128, A_img - A_cpx_64, A_cpx_64]], augm='np.abs', cbar=True)

B = harray.to_stacked(A_img, 'cartesian', stack_ax=-1)
B_img = harray.to_complex(B)
B_float_64 = harray.to_complex(B.astype(np.float64)).astype(np.complex128)
B_float_32 = harray.to_complex(B.astype(np.float32)).astype(np.complex128)
B_float_16 = harray.to_complex(B.astype(np.float16)).astype(np.complex128)

hplotc.ListPlot([[B_img - B_float_64, B_float_64,
                  B_img - B_float_32, B_float_32,
                  B_img - B_float_16, B_float_16]], augm='np.abs', cbar=True)

B_int_8 = harray.to_complex((np.iinfo(np.int8).max * B).astype(np.int8)).astype(np.complex128)
B_int_16 = harray.to_complex((np.iinfo(np.int16).max * B).astype(np.int16)).astype(np.complex128)
B_int_32 = harray.to_complex((np.iinfo(np.int32).max * B).astype(np.int32)).astype(np.complex128)
B_int_64 = harray.to_complex((np.iinfo(np.int64).max * B).astype(np.int64)).astype(np.complex128)

hplotc.ListPlot([[(np.iinfo(np.int8).max * B_img) - B_int_8, B_int_8,
                  (np.iinfo(np.int16).max * B_img) - B_int_16, B_int_16,
                  (np.iinfo(np.int32).max * B_img) - B_int_32, B_int_32,
                  (np.iinfo(np.int64).max * B_img) - B_int_64, B_int_64]], augm='np.abs', cbar=True)
