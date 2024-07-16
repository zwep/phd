"""
Make some radial images...

Some are also made in the `test_on_cardiac_acquisition`
"""


import reconstruction.ReadCpx as read_cpx
import helper.plot_class as hplotc
import numpy as np
import sigpy.mri
import sigpy
import helper.array_transf as harray
from skimage.util import img_as_ubyte
import skimage.transform as sktransf
import imageio
import scipy.io
import os

# Define files [TRANSVERSE]
target_dir = '/home/bugger/Documents/presentaties/RF_meetings/RF_meeting_20210915'
base_path = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac/V9_19526'
radial_file_untriggered = 'v9_02052021_0931525_9_2_transradial_no_trigV4.mat'
radial_file_cardiac_resp_triggered = 'v9_02052021_0929572_7_2_transradialfastV4.mat'
radial_file_high_time = 'v9_02052021_0930471_8_2_transradialfast_high_timeV4.mat'
cartesian_file_cardiac_resp_triggered = 'v9_02052021_0928462_6_2_cine1slicer2_traV4.mat'

# Load files
# Mat file
radial_array_untriggered = scipy.io.loadmat(os.path.join(base_path, radial_file_untriggered))['reconstructed_data']
radial_array_cardiac_resp_triggered = scipy.io.loadmat(os.path.join(base_path, radial_file_cardiac_resp_triggered))['reconstructed_data']
radial_array_high_time = scipy.io.loadmat(os.path.join(base_path, radial_file_high_time))['reconstructed_data']
cartesian_array_cardiac_resp_triggered = scipy.io.loadmat(os.path.join(base_path, cartesian_file_cardiac_resp_triggered))['reconstructed_data']

"""
Visualize one cardiac phase for all examples..
"""

plot_array = [radial_array_untriggered, radial_array_cardiac_resp_triggered, radial_array_high_time, cartesian_array_cardiac_resp_triggered]
sel_slice_array = [40, 4, 12, 20]
plot_array = [np.squeeze(np.take(x, y, axis=-1)) for x, y in zip(plot_array, sel_slice_array)]
hplotc.ListPlot([plot_array], augm='np.abs', subtitle=[['Radial - No triggering', 'Radial - triggered: fs', 'Radial - triggered: us', 'Cartesian - triggered']],
                ax_off=True)

"""
Undersample with 30 % the cartesian one
"""

fully_sampled_image = np.squeeze(np.moveaxis(cartesian_array_cardiac_resp_triggered, -1, 0))
n_card = fully_sampled_image.shape[0]
img_shape = fully_sampled_image.shape[-2:]
n_spokes = n_points = max(img_shape)
time_undersampled = []
p_sample = 60
n_undersample = int((p_sample / 100) * n_spokes)
# Define trajectory..
trajectory_radial = sigpy.mri.radial(coord_shape=(n_spokes, n_points, 2), img_shape=img_shape)
trajectory_radial = trajectory_radial.reshape(-1, 2)
dcf = np.sqrt(trajectory_radial[:, 0] ** 2 + trajectory_radial[:, 1] ** 2)

# Define undersampled trajectory, the same for ALL the coils
undersampled_trajectory = np.array(np.split(trajectory_radial, n_spokes))
random_lines = np.random.choice(range(n_spokes), size=(n_spokes - n_undersample), replace=False)
undersampled_trajectory[random_lines] = None
undersampled_trajectory = undersampled_trajectory.reshape(-1, 2)
ovs = 1.25
width = 6
for i_card in range(n_card):
    temp_kspace = sigpy.nufft(fully_sampled_image[i_card], coord=undersampled_trajectory, width=width, oversamp=ovs)
    temp_img = sigpy.nufft_adjoint(temp_kspace * dcf, coord=undersampled_trajectory, oshape=img_shape, width=width, oversamp=ovs)

    time_undersampled.append(np.abs(temp_img))

time_undersampled = [harray.scale_minmax(x, is_complex=True) for x in time_undersampled]
us_img_ubyte = img_as_ubyte(np.array(time_undersampled))
us_img_ubyte = sktransf.resize(us_img_ubyte, (n_card, 256, 256), preserve_range=True).astype(np.uint8)
target_file = os.path.join(target_dir, 'cart_60_perc_same_undersampling_example.gif')
imageio.mimsave(target_file, us_img_ubyte, duration=5/n_card)


"""
Visualize fully sampled cardiac cine and undersmpled (RADIAL)
"""
radial_array_cardiac_resp_triggered = scipy.io.loadmat(os.path.join(base_path, radial_file_cardiac_resp_triggered))['reconstructed_data']
radial_array_high_time = scipy.io.loadmat(os.path.join(base_path, radial_file_high_time))['reconstructed_data']
radial_array_untriggered = scipy.io.loadmat(os.path.join(base_path, radial_file_untriggered))['reconstructed_data']

radial_array_high_time = np.abs(np.squeeze(np.moveaxis(radial_array_high_time, -1, 0)))
hplotc.ListPlot(radial_array_high_time[0], cbar=True)
radial_array_high_time[radial_array_high_time > 90] = 90
radial_array_high_time = harray.scale_minmax(radial_array_high_time)
radial_array_high_time = img_as_ubyte(radial_array_high_time)

radial_array_cardiac_resp_triggered = np.abs(np.squeeze(np.moveaxis(radial_array_cardiac_resp_triggered, -1, 0)))
radial_array_cardiac_resp_triggered = harray.scale_minmax(radial_array_cardiac_resp_triggered)
radial_array_cardiac_resp_triggered = img_as_ubyte(radial_array_cardiac_resp_triggered)

radial_array_untriggered = np.abs(np.squeeze(np.moveaxis(radial_array_untriggered, -1, 0)))
hplotc.ListPlot(radial_array_untriggered[0], cbar=True)
radial_array_untriggered[radial_array_untriggered > 60] = 60
radial_array_untriggered = harray.scale_minmax(radial_array_untriggered)
radial_array_untriggered = img_as_ubyte(radial_array_untriggered)

# hplotc.SlidingPlot(radial_array_high_time)
target_file = os.path.join(target_dir, 'undersampled_radial_no_trig.gif')
imageio.mimsave(target_file, radial_array_untriggered, duration=5/(radial_array_high_time.shape[0]))

# hplotc.SlidingPlot(radial_array_cardiac_resp_triggered)
target_file = os.path.join(target_dir, 'fullysampled_radial.gif')
imageio.mimsave(target_file, radial_array_cardiac_resp_triggered, duration=5/(radial_array_cardiac_resp_triggered.shape[0]))

"""
Create on single slice as example... AND SAVE CARTESIAN AS GIF
"""
cartesian_array_cardiac_resp_triggered = scipy.io.loadmat(os.path.join(base_path, cartesian_file_cardiac_resp_triggered))['reconstructed_data']
cartesian_array_cardiac_resp_triggered = np.abs(np.squeeze(np.moveaxis(cartesian_array_cardiac_resp_triggered, -1, 0)))
cartesian_array_cardiac_resp_triggered[cartesian_array_cardiac_resp_triggered>815]=815
cartesian_array_cardiac_resp_triggered = harray.scale_minmax(cartesian_array_cardiac_resp_triggered)
cartesian_array_cardiac_resp_triggered = img_as_ubyte(cartesian_array_cardiac_resp_triggered)
n_card = cartesian_array_cardiac_resp_triggered.shape[0]
cartesian_array_cardiac_resp_triggered_resize = sktransf.resize(cartesian_array_cardiac_resp_triggered, (n_card, 256, 256), preserve_range=True).astype(np.uint8)

# This can be saved...
hplotc.ListPlot([[radial_array_cardiac_resp_triggered[4], cartesian_array_cardiac_resp_triggered[15]]], ax_off=True)

target_file = os.path.join(target_dir, 'cartesian_acquisition.gif')
imageio.mimsave(target_file, cartesian_array_cardiac_resp_triggered_resize, duration=5/(n_card))



