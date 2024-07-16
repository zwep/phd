
import os
import numpy as np
import torch
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.misc as hmisc
import objective.undersampled_recon.executor_undersampled_recon as executor
import helper.array_transf as harray
from skimage.util import img_as_ubyte, img_as_uint
import skimage.transform as sktransf
import imageio
import reconstruction.ReadCpx as read_cpx
import reconstruction.ReadRec as read_rec



"""
Load some cpx data
"""

output_path = '/home/bugger/Documents/presentaties/RF_meetings/RF_meeting_20210915'
patient_dir = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_13/V9_17069'

# Cpx file of carteisna one is of course... SENSE'd
groundtruth_file = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac/V9_17069/transverse/v9_13022021_1343308_5_3_cine1slicer2_traV4.npy'
radial_fs_file = os.path.join(patient_dir, 'v9_13022021_1345094_6_3_transradialfastV4.cpx')
radial_us_file = os.path.join(patient_dir, 'v9_13022021_1346189_7_3_transradialfast_high_timeV4.cpx')
radial_no_trigger_file = os.path.join(patient_dir, 'v9_13022021_1347235_8_3_transradial_no_trigV4.cpx')

# Another batch
# patient_dir = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_06/V9_16936'
# groundtruth_file = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac/V9_16936/transverse/v9_06022021_1220584_5_3_cine1slicer2_traV4.npy'
# radial_fs_file = os.path.join(patient_dir, 'v9_06022021_1223187_6_3_transradialfastV4.cpx')
# radial_us_file = os.path.join(patient_dir, 'v9_06022021_1224407_7_3_transradialfast_high_timeV4.cpx')
# radial_no_trigger_file = os.path.join(patient_dir, 'v9_06022021_1226370_8_3_transradial_no_trigV4.cpx')


# Load cartesian ground truth..
# cpx_cartesian = scipy.io.loadmat(groundtruth_file)['reconstructed_data']
# cpx_cartesian = np.moveaxis(np.squeeze(cpx_cartesian), -1, 0)
cpx_cartesian = np.load(groundtruth_file)
cpx_cartesian = np.squeeze(cpx_cartesian)
hplotc.SlidingPlot(cpx_cartesian)

# Convert to GIF
n_card = cpx_cartesian.shape[0]
image_to_gif_array = np.abs(cpx_cartesian)
hplotc.ListPlot(image_to_gif_array[0, 100:400], cbar=True)
hmisc.convert_image_to_gif(image_to_gif_array[:, 100:400],
                     output_path=os.path.join(output_path, 'cart_fully_sampled_acq.gif'),
                     n_card=n_card,
                     nx=128, ny=256)

# Load fully sampled radial
cpx_obj = read_cpx.ReadCpx(radial_fs_file)
cpx_radial_fs = np.squeeze(cpx_obj.get_cpx_img())
# cpx_radial_fs = scipy.io.loadmat(radial_fs_file)['reconstructed_data']
# cpx_radial_fs = np.moveaxis(np.squeeze(cpx_radial_fs), -1, 0)
hplotc.SlidingPlot(cpx_radial_fs.sum(axis=0)[:, ::-1, ::-1])

# Convert to GIF
n_card = cpx_radial_fs.shape[1]
image_to_gif_array = np.abs(cpx_radial_fs.sum(axis=0)[:, ::-1, ::-1])
# hplotc.ListPlot(image_to_gif_array[0], cbar=True)
image_to_gif_array[image_to_gif_array > 57000] = 57000
hmisc.convert_image_to_gif(image_to_gif_array,
                     output_path=os.path.join(output_path, 'radial_fully_sampled_acq.gif'),
                     n_card=n_card,
                     nx=256, ny=256)

# Load under sampled radial..
cpx_obj = read_cpx.ReadCpx(radial_us_file)
cpx_radial_us = cpx_obj.get_cpx_img()
cpx_radial_us = np.squeeze(cpx_radial_us)
hplotc.SlidingPlot(cpx_radial_us.sum(axis=0)[:, ::-1, ::-1])

n_card = cpx_radial_us.shape[1]
image_to_gif_array = np.abs(cpx_radial_us.sum(axis=0)[:, ::-1, ::-1])
image_to_gif_array[image_to_gif_array > 70000] = 70000
hmisc.convert_image_to_gif(image_to_gif_array,
                     output_path=os.path.join(output_path, 'radial_under_sampled_acq.gif'),
                     n_card=n_card,
                     nx=256, ny=256)


"""
Check the effect of undersampling...
"""


# # DIrty hack to undersample this fully sampled example to 20% orso
import sigpy
import sigpy.mri
# We could select only the last 8 coils if we want...
# Ideally we want to have a subset of this or something like that.
# loaded_array = loaded_array[-8:]
n_chan, n_card = cpx_radial_us.shape[:2]
img_shape = cpx_radial_us.shape[-2:]
p_undersample = 20
# This is almost the same over all the images because of a weird circle...
x_size, y_size = img_shape
x_range = np.linspace(-x_size // 2, x_size // 2, x_size)
y_range = np.linspace(-y_size // 2, y_size // 2, y_size)
X, Y = np.meshgrid(x_range, y_range)
mask_array = torch.from_numpy(np.sqrt(X ** 2 + Y ** 2) <= x_size // 2)[None]

# Define dimensions radial spokes
n_spokes = n_points = max(img_shape)
n_undersample = int((p_undersample / 100) * n_spokes)
# Define trajectory..
trajectory_radial = sigpy.mri.radial(coord_shape=(n_spokes, n_points, 2), img_shape=img_shape)
trajectory_radial = trajectory_radial.reshape(-1, 2)
# We might remove this one..? Because it is so generic and repetitive over all the spokes
dcf = np.sqrt(trajectory_radial[:, 0] ** 2 + trajectory_radial[:, 1] ** 2)

# Define undersampled trajectory, the same for ALL the coils
undersampled_trajectory = np.array(np.split(trajectory_radial, n_spokes))
# We selecteren hier indices van de lijnen die we WEG willen hebben
# Dus bij undersampled trajectory worden er n - n_undersampled lijnen op 'null' gezet
# Zo behouden n_undersampled lijnen hun data
random_lines = np.random.choice(range(n_spokes), size=(n_spokes - n_undersample), replace=False)
undersampled_trajectory[random_lines] = None
undersampled_trajectory = undersampled_trajectory.reshape(-1, 2)


def undersample_img(card_array, width, ovs, traj, dcf, img_shape):
    input_array = []
    for i_coil in card_array[-8:]:
        temp_kspace = sigpy.nufft(i_coil, coord=traj, width=width, oversamp=ovs)
        temp_img = sigpy.nufft_adjoint(temp_kspace * dcf, coord=traj, oshape=img_shape,
                                       width=width, oversamp=ovs)
        input_array.append(temp_img)

    input_array = np.array(input_array)
    return input_array

img_to_undersample = np.swapaxes(cpx_radial_fs, 0, 1)
undersampled_array = [undersample_img(x, width=6, ovs=1.25, traj=undersampled_trajectory, img_shape=img_shape, dcf=dcf) for x in img_to_undersample]
undersampled_array = np.array(undersampled_array)
n_card = cpx_radial_us.shape[0]
image_to_gif_array = np.abs(undersampled_array.sum(axis=1)[:, ::-1, ::-1])
hplotc.ListPlot(image_to_gif_array[0], cbar=True)
image_to_gif_array[image_to_gif_array > 2e6] = 2e6
hmisc.convert_image_to_gif(image_to_gif_array,
                     output_path=os.path.join(output_path, 'artf_undersampled_acq.gif'),
                     n_card=n_card,
                     nx=256, ny=256)
# Store as GIF..
# # /end


