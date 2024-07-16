import os
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=6
import helper.array_transf as harray
import biasfield_algorithms.N4ITK as model_n4itk
import h5py
from skimage.util import img_as_ubyte, img_as_uint
import helper.misc as hmisc
import numpy as np
import nibabel

from objective_configuration.inhomog_removal import get_path_dict
"""
Load the 3T data

Correct 3T data with N4

Store this correction
"""

dn4 = '/home/sharreve/local_scratch/model_run/selected_inhomog_removal_models/n4'
path_dict, _ = get_path_dict(dn4)

dsource = path_dict['synthetic']['dimage']
dmask = path_dict['synthetic']['dmask']
ddest = os.path.join(path_dict['synthetic']['dpred'], 'pred')

if not os.path.isdir(ddest):
    os.makedirs(ddest)

file_list = os.listdir(dsource)
for i_file in sorted(file_list):
    ext_file = hmisc.get_ext(i_file)
    input_file = os.path.join(dsource, i_file)
    mask_file = os.path.join(dmask, i_file)
    storage_file = os.path.join(ddest, i_file)
    print('Processing ', input_file)
    print('Storing to ', storage_file)
    input_array = hmisc.load_array(input_file)
    mask_array = hmisc.load_array(mask_file)
    if 'nii' in ext_file:
        input_array = input_array.T[:, ::-1, ::-1]
        mask_array = mask_array.T[:, ::-1, ::-1]
    n_slice = input_array.shape[0]
    corrected_image = []
    for sel_slice in range(n_slice):
        print('Performing slice ', sel_slice, '/', n_slice, end='\r')
        input_array_slice = input_array[sel_slice]
        mask_array_slice = mask_array[sel_slice]
        # Scale it from uint to float 0..1
        input_array_slice = harray.scale_minmax(input_array_slice)
        output_n4itk = model_n4itk.get_n4itk(input_array_slice, mask=mask_array_slice)
        output_n4itk = harray.scale_minmax(output_n4itk)
        corrected_image.append(output_n4itk)
    corrected_image = np.array(corrected_image)
    corrected_image = harray.scale_minmax(corrected_image)
    corrected_image = img_as_ubyte(corrected_image)
    nibabel_obj = nibabel.Nifti1Image(corrected_image.T[::-1, ::-1, :], np.eye(4))
    nibabel.save(nibabel_obj, storage_file)
