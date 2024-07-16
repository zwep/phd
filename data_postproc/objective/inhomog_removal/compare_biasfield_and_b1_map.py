"""
Maybe we can extract B1...?
"""
import nibabel
import os
import reconstruction.ReadCpx as read_cpx
import helper.plot_class as hplotc
import pydicom
import numpy as np

dbiasf = '/media/bugger/MyBook/data/paper/inhomog_removal/volunteer/single_homogeneous/biasfield/pr_06012021_1653458_15_3_t2wV4.nii.gz'
biasfimg = nibabel.load(dbiasf).get_fdata().T[:, ::-1, ::-1]

input_dir = '/media/bugger/MyBook/data/7T_scan/prostate/2021_01_06/pr_16289'
input_dir_dicom = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/2021_01_06/pr_16289'
b1_dicom_dir = [os.path.join(input_dir_dicom, x) for x in os.listdir(input_dir_dicom) if 'B1map' in x]
b1_dicom_files = [os.path.join(x, 'DICOM/IM_0002') for x in b1_dicom_dir]

survey_scan_file = os.path.join(input_dir, 'pr_06012021_1639155_7_2_surveyisoV4.cpx')
b1_shim_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if 'b1shim' in x and x.endswith('cpx')]
b1_map_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if 'b1map' in x and x.endswith('cpx')]
t2_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if 't2' in x and x.endswith('cpx')]

b1_map_array = [np.squeeze(read_cpx.ReadCpx(x).get_cpx_img()) for x in b1_map_files]
b1_dicom_obj = [pydicom.read_file(x) for x in b1_dicom_files]
b1_dicom_array = np.array([x.pixel_array for x in b1_dicom_obj])
# Scale B1 map to flip angle and test sin()^3 mapping
b1_dicom_array.shape
b1_sel = b1_dicom_array[:, 3]

import helper.array_transf as harray

ww = harray.scale_minmax(biasfimg[0])
zz = np.arcsin(harray.smooth_image(ww, n_kernel=32) ** (1/3))
hplotc.ListPlot([zz, b1_sel[0]])
