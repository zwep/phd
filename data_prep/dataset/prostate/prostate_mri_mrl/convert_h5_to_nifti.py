import numpy as np
from skimage.util import img_as_ubyte, img_as_uint, img_as_int
import helper.array_transf as harray
import helper.misc as hmisc
import os
import nibabel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-source', type=str)
parser.add_argument('-target', type=str)

p_args = parser.parse_args()
dorig = p_args.source
ddest = p_args.target

if not os.path.isdir(ddest):
    os.makedirs(ddest)

file_list = [x for x in os.listdir(dorig) if x.endswith('h5')]

for sel_file in file_list:
    file_name, ext = os.path.splitext(sel_file)
    file_path = os.path.join(dorig, sel_file)
    dest_file_path = os.path.join(ddest, file_name + '.nii.gz')
    temp_A = hmisc.load_array(file_path)
    temp_A = temp_A.T[::-1, ::-1]
    temp_A = harray.scale_minmax(temp_A, axis=(0, 1))
    temp_A = img_as_ubyte(temp_A)
    nibabel_obj = nibabel.Nifti1Image(temp_A, np.eye(4))
    print(f'Convert {file_path}')
    print(f'\t To {dest_file_path}')
    nibabel.save(nibabel_obj, dest_file_path)
