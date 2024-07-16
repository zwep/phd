import numpy as np
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
# dorig = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/image'
# ddest = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/image_nifti'

file_list = [x for x in os.listdir(dorig) if x.endswith('npy')]
for sel_file in file_list:
    file_name, ext = os.path.splitext(sel_file)
    file_path = os.path.join(dorig, sel_file)
    dest_file_path = os.path.join(ddest, file_name + '.nii.gz')
    temp_A = np.load(file_path)
    if temp_A.ndim == 2:
        temp_A = temp_A[None]
    temp_A = temp_A.T[::-1, ::-1]
    # This can be done prettier but I bet it works.
    # Nifti cant deal with bools
    if temp_A.dtype == 'bool':
        temp_A = temp_A.astype(int)
    nibabel_obj = nibabel.Nifti1Image(temp_A, np.eye(4))
    print(f'Convert {file_path}')
    print(f'\t To {dest_file_path}')
    nibabel.save(nibabel_obj, dest_file_path)
