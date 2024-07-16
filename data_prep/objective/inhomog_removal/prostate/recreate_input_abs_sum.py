import nibabel
from skimage.util import img_as_ubyte, img_as_uint, img_as_int
import numpy as np
import helper.misc as hmisc
import helper.array_transf as harray
import os

"""
Since Ive added noise to the input data the input abs sum images were not updated

Which means that the single channel models have it easier.


"""

dsource = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/input_nifti'
ddest = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/input_abs_sum_nifti'

file_list = os.listdir(dsource)
file_list = [x for x in file_list if x.endswith('nii.gz')]

for i_file in file_list:
    source_file = os.path.join(dsource, i_file)
    dest_file = os.path.join(ddest, i_file)
    source_array = hmisc.load_array(source_file)
    source_cpx_array = source_array[:, :, :, :, 0] + 1j * source_array[:, :, :, :, 0]
    sos_source_array = np.sqrt(np.sum(np.abs(source_cpx_array) ** 2, axis=-1))
    sos_source_array = harray.scale_minmax(sos_source_array, axis=(0, 1))
    sos_source_array = img_as_ubyte(sos_source_array)
    new_obj = nibabel.Nifti1Image(sos_source_array, np.eye(4))
    nibabel.save(new_obj, dest_file)
    print('Written ', dest_file)