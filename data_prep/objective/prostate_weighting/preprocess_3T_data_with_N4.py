import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import helper.array_transf as harray
import biasfield_algorithms.N4ITK as model_n4itk
import h5py
from skimage.util import img_as_ubyte, img_as_uint
import helper.misc as hmisc
import numpy as np


"""
Load the 3T data

Correct 3T data with N4

Store this correction
"""

for data_type in ['train', 'test', 'validation']:
    data_dir = '/local_scratch/sharreve/mri_data/prostate_weighting_h5'
    target_dir = os.path.join(data_dir, data_type, 'target')
    mask_dir = os.path.join(data_dir, data_type, 'mask')
    file_list = os.listdir(target_dir)

    storage_dir = os.path.join(data_dir, data_type, 'target_corrected_N4')
    if not os.path.isdir(storage_dir):
        os.makedirs(storage_dir)

    for i_file in sorted(file_list):
        file_name, ext = os.path.splitext(i_file)
        file_name_mask = file_name + '_target' + ext
        input_file = os.path.join(target_dir, i_file)
        mask_file = os.path.join(mask_dir, file_name_mask)
        storage_file = os.path.join(storage_dir, i_file)
        print('Processing ', input_file)
        print('Storing to ', storage_file)
        with h5py.File(input_file, 'r') as f:
            target_array = np.array(f['data'])
        with h5py.File(mask_file, 'r') as f:
            mask_array = np.array(f['data'])

        n_slice = target_array.shape[0]
        print()
        corrected_3t_image = []
        for sel_slice in range(n_slice):
            print('Performing slice ', sel_slice, '/', n_slice, end='\r')
            # input_array_slice = input_array[sel_slice]
            target_array_slice = target_array[sel_slice]
            mask_array_slice = mask_array[sel_slice]

            # Scale it from uint to float 0..1
            target_array_slice = harray.scale_minmax(target_array_slice)
            output_n4itk = model_n4itk.get_n4itk(target_array_slice, mask=mask_array_slice)
            corrected_image = harray.scale_minmax(output_n4itk)
            corrected_3t_image.append(corrected_image)
        corrected_3t_image = np.array(corrected_3t_image) * mask_array
        corrected_3t_image = harray.scale_minmax(corrected_3t_image)
        with h5py.File(storage_file, 'w') as f:
            f.create_dataset('data', data=img_as_ubyte(corrected_3t_image))


