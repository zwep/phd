from skimage import img_as_int
import helper.misc as hmisc
import helper.plot_class as hplotc
import skimage.util.dtype as skdtype
import helper.array_transf as harray
import h5py
import numpy as np

"""
Again.. not sure which storage method is best...
"""

# This file was 3.6Gb....
ddata = '/local_scratch/sharreve/mri_data/registrated_h5/train/input/Duke_to_10_MR_20210107_0002_transversal.h5'
with h5py.File(ddata, 'r') as f:
    res = np.array(f['data'])

b1m_array_registered = res[0]
b1m_array_registered_cpx = b1m_array_registered[0] + 1j * b1m_array_registered[1]
plot_obj = hplotc.ListPlot([b1m_array_registered_cpx], augm='np.real', cbar=True)
plot_obj.figure.savefig('/home/sharreve/local_scratch/first_resulat_registration.png')

store_cpx_path = '/home/sharreve/local_scratch/test_storage_cpx_complex64.h5'
store_cpx_int16_path = '/home/sharreve/local_scratch/test_storage_cpx_int16.h5'
store_stacked_int16_path = '/home/sharreve/local_scratch/test_storage_stacked_int16.h5'
store_stacked_int8_path = '/home/sharreve/local_scratch/test_storage_stacked_int8.h5'

# Storage methods...
with h5py.File(store_cpx_path, 'w') as h5_obj:
    h5_obj.create_dataset('data', data=b1m_array_registered.astype(np.complex64))

b1m_scaled = harray.scale_minmax(b1m_array_registered, is_complex=True)
int_complex = img_as_int(b1m_scaled.real) + 1j * img_as_int(b1m_scaled.imag)
with h5py.File(store_cpx_int16_path, 'w') as h5_obj:
    h5_obj.create_dataset('data', data=int_complex)

int_stacked = np.stack([img_as_int(b1m_scaled.real), img_as_int(b1m_scaled.imag)], axis=1)
with h5py.File(store_stacked_int16_path, 'w') as h5_obj:
    h5_obj.create_dataset('data', data=int_stacked)

int8_stacked = np.stack([skdtype._convert(b1m_scaled.real, np.int8), skdtype._convert(b1m_scaled.imag, np.int8)], axis=1)
with h5py.File(store_stacked_int8_path, 'w') as h5_obj:
    h5_obj.create_dataset('data', data=int8_stacked)

# Visualizations...

cpx_array = hmisc.load_array(store_cpx_path)
cpx_int16_array = hmisc.load_array(store_cpx_int16_path)
cpx_stack16_array = hmisc.load_array(store_stacked_int16_path)
cpx_stack16_array = cpx_stack16_array[:, 0] + 1j *cpx_stack16_array[:, 1]
cpx_stack8_array = hmisc.load_array(store_stacked_int8_path)
cpx_stack8_array = cpx_stack8_array[:, 0] + 1j *cpx_stack8_array[:, 1]
plot_obj = hplotc.ListPlot([cpx_array, cpx_int16_array, cpx_stack16_array,cpx_stack8_array], augm='np.real', cbar=True)
plot_obj.figure.savefig('/home/sharreve/local_scratch/visualiziation_storage_effect.png')
