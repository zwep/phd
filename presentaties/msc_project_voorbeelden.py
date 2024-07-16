"""
Hier maken we wat platjes voor de verschillende projecten die ik heb bedacht
"""

import os
import numpy as np
import helper.plot_class as hplotc
import scipy.io

dflavio = '/media/bugger/MyBook/data/simulated/b1p_b1m_flavio_mat'
sel_file = [x for x in os.listdir(dflavio) if x.endswith('mat')][0]

flavio_file = os.path.join(dflavio, sel_file)
mat_file = scipy.io.loadmat(flavio_file)['Model']
hplotc.ListPlot(mat_file['Body'][0][0])
b1_minus = mat_file['B1minus'][0][0]
b1_minus = np.moveaxis(b1_minus, -1, 0)
b1_plus = mat_file['B1plus'][0][0]
b1_plus = np.moveaxis(b1_plus, -1, 0)
hplotc.ListPlot([b1_minus, b1_plus], augm='np.abs', ax_off=True, vmin=(0, 0.3 * np.abs(b1_plus).max()),
                subtitle=[['b1-', '', '', '', '', '', '', '', ''], ['b1+', '', '', '', '', '', '', '', '']])
hplotc.ListPlot([b1_minus, b1_plus], augm='np.angle', ax_off=True,
                subtitle=[['b1-', '', '', '', '', '', '', '', ''], ['b1+', '', '', '', '', '', '', '', '']])

# Get some 1.5 T image...
import nrrd
dprostate = '/media/bugger/MyBook/data/prostatemriimagedatabase_request'
sel_file = [x for x in os.listdir(dprostate) if x.endswith('nrrd')][17]
prostate_file = os.path.join(dprostate, sel_file)
img_array, nrrd_param = nrrd.read(prostate_file)
img_array = np.moveaxis(img_array, -1, 0)
img_array = np.rot90(img_array, k=3, axes=(-2, -1))
hplotc.ListPlot(img_array[41])


"""
Cardiac segmentation stuff
"""
import nibabel
# Example of one file...
LA_cine_file = '/media/bugger/MyBook/data/m&m/MnM-2/training/001/001_LA_CINE.nii.gz'
LA_ed_file = '/media/bugger/MyBook/data/m&m/MnM-2/training/001/001_LA_ED.nii.gz'
LA_ed_gt_file = '/media/bugger/MyBook/data/m&m/MnM-2/training/001/001_LA_ED_gt.nii.gz'

nib_obj = nibabel.load(LA_cine_file)
cine_array = nib_obj.get_fdata()

nib_obj = nibabel.load(LA_ed_file)
LA_ed_array = nib_obj.get_fdata()

nib_obj = nibabel.load(LA_ed_gt_file)
LA_ed_gt_array = nib_obj.get_fdata()

hplotc.ListPlot([[LA_ed_gt_array[:, :, 0], LA_ed_array[:, :, 0]]], ax_off=True)


"""
Example of radial undersampled file..
"""

import reconstruction.ReadCpx as read_cpx
import numpy as np
dradial = '/media/bugger/MyBook/data/7T_data/cartesian_radial_dataset_transverse/train/input/v9_02052021_0925327.npy'
dcartesian = '/media/bugger/MyBook/data/7T_data/cartesian_radial_dataset_transverse/train/target/v9_02052021_0925327.npy'
radial_array = np.load(dradial)
cart_array = np.load(dcartesian)
hplotc.ListPlot([[radial_array.sum(axis=0)[0], cart_array[0]]], augm='np.abs', ax_off=True)
