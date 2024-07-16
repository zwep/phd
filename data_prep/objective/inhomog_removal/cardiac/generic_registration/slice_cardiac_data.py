import os
import re
import scipy.ndimage.interpolation as scint
import numpy as np
import helper.plot_class as hplotc
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import helper.array_transf as harray
import scipy.io
import helper.misc as hmisc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import h5py
import scipy.ndimage


class AngulatedSlice:
    """
    This thing is used to get an idea on how we can rotate a slice and the effect it has on a 3D array
    """
    def __init__(self, slicing_image=None, **kwargs):
        """
        Is able to easily plot n-dimensional data, where the last two are used to plt.imshow()
        The others become Slider axes that can be issued.
        """

        self.slicing_image = slicing_image

        self.fig = plt.figure()
        self.ax3d = self.fig.add_subplot(1, 2, 1, projection='3d')
        # Second subplot
        self.ax2d = self.fig.add_subplot(1, 2, 2)
        if slicing_image is not None:
            # Assuming slicing image is a 3d image.....
            self.ax2d_imshow = self.ax2d.imshow(slicing_image[0])

        self.axcolor = 'lightgoldenrodyellow'
        self.delta_ax = 0.05

        self.x_size_box, self.x_angle_slider, self.x_offset_slider = self.get_group(init_ax_pos=0.25, label='x')
        self.y_size_box, self.y_angle_slider, self.y_offset_slider = self.get_group(init_ax_pos=0.55, label='y')
        self.z_size_box, self.z_angle_slider, self.z_offset_slider = self.get_group(init_ax_pos=0.75, label='z')

        # self.button_ax = plt.axes([0.05, 0.9, 0.2, 0.04])
        # self.calculate_button = Button(ax=self.button_ax, label='calculate')
        # self.calculate_button.connect_event('button_press_event', self.press_event_handler)

        self.slice_button_ax = plt.axes([0.05, 0.95, 0.2, 0.04])
        self.slice_button = Button(ax=self.slice_button_ax, label='Update')
        self.slice_button.connect_event('button_press_event', self.press_event_handler)

        # Here we store the obtained plane...
        self.acq_plane_rot = None
        self.update(None)

    def get_group(self, init_ax_pos, label='x'):
        if self.slicing_image is not None:
            if label == 'x':
                initial_size = self.slicing_image.shape[0]
            elif label == 'y':
                initial_size = self.slicing_image.shape[1]
            elif label == 'z':
                initial_size = self.slicing_image.shape[2]
        else:
            initial_size = 100

        # Set initial dimension box...
        size_axis = plt.axes([0.05, init_ax_pos, 0.1, 0.025])
        size_box = TextBox(ax=size_axis, label=f"{label}_size", initial=str(initial_size))
        # size_box.on_text_change(self.update)

        # Rotate slider
        angle_axis = plt.axes([0.05, init_ax_pos + self.delta_ax, 0.1, 0.025])
        angle_slider = Slider(ax=angle_axis, label=f"{label}_angle", valmin=-90, valmax=90, valinit=0)
        angle_slider.on_changed(self.press_event_handler)

        offset_axis = plt.axes([0.05, init_ax_pos + 2*self.delta_ax, 0.1, 0.025])
        offset_slider = Slider(ax=offset_axis, label=f"{label}_offset", valmin=-100, valmax=100, valinit=0)
        offset_slider.on_changed(self.press_event_handler)

        return size_box, angle_slider, offset_slider

    def update(self, value):
        max_x = int(self.x_size_box.text)
        max_y = int(self.y_size_box.text)
        max_z = int(self.z_size_box.text)

        max_list = [max_z, max_y, max_x]
        mid_point = np.array(self.slicing_image.shape)//2
        mid_point[0] = 0
        angle_list = [float(self.z_angle_slider.val), float(self.y_angle_slider.val), float(self.x_angle_slider.val)]
        offset_list = [int(self.z_offset_slider.val), int(self.y_offset_slider.val), int(self.x_offset_slider.val)]
        self.acq_plane_rot = get_affine_slice_coords(max_list=max_list, angle_list=angle_list, offset_list=offset_list, delta_list=[0,0,0], mid_point=mid_point)

        self.ax3d.cla()
        self.ax3d.scatter(self.acq_plane_rot[::10, 0], self.acq_plane_rot[::10, 1], self.acq_plane_rot[::10, 2], c='b', label='Non rotated plane')
        self.ax3d.axes.set_xlim3d(left=0, right=max_x)
        self.ax3d.axes.set_ylim3d(bottom=0, top=max_y)
        self.ax3d.axes.set_zlim3d(bottom=0, top=max_z)
        self.fig.canvas.draw()
        # self.fig.batch_update()

    def press_event_handler(self, event):
        #if event.inaxes == self.button_ax:
            # print('Calculating...')
            # ny, nx = int(self.y_size_box.text), int(self.x_size_box.text)
            #
            # temp_real = scipy.ndimage.map_coordinates(self.slicing_image, self.acq_plane_rot.T, cval=-1).reshape((ny, nx))
            # print('...Done!', end='\n\n')
            # self.ax2d_imshow.set_data(temp_real)
            # min_plot = np.min(temp_real)
            # max_plot = np.max(temp_real)
            # varray = [min_plot, max_plot]
            # self.ax2d_imshow.set_clim(varray)
        #el..
        #if event.inaxes == self.slice_button_ax:
        self.update(None)
        print('Calculating...')
        ny, nx = int(self.y_size_box.text), int(self.x_size_box.text)

        temp_real = scipy.ndimage.map_coordinates(np.copy(self.slicing_image), self.acq_plane_rot.T, cval=-1).reshape((ny, nx))
        print('...Done!', end='\n\n')
        self.ax2d_imshow.set_data(temp_real)
        min_plot = np.min(temp_real)
        max_plot = np.max(temp_real)
        varray = [min_plot, max_plot]
        self.ax2d_imshow.set_clim(varray)


# def slice_operator(x, angle_operator):
#     # Used to simply rotate the array and get a couple of slices
#     angle1 = angle_operator['angle1']
#     angle2 = angle_operator['angle2']
#     angle3 = angle_operator['angle3']
#     slice_min = angle_operator['slice_min']
#     slice_max = angle_operator['slice_max']
#     # In version 1.6.0 - scipy rotate can deal with complex images
#     temp1 = scipy.ndimage.rotate(x, angle=angle1, axes=(-2, -1))
#     temp1 = np.moveaxis(temp1, -1, 0)
#     temp2 = scipy.ndimage.rotate(temp1, angle=angle2, axes=(-2, -1))
#     temp2 = np.moveaxis(temp2, -2, 0)
#     temp3 = scipy.ndimage.rotate(temp2, angle=angle3, axes=(-2, -1))
#     temp3 = np.swapaxes(temp3, -2, -1)
#     selected_slices = temp3[slice_min:slice_max]
#     return selected_slices


def rotate_point(x, rot_mat, point):
    return rot_mat @ (x - point) + point


def get_affine_slice_coords(max_list, delta_list, offset_list, angle_list, n_points_y=None,
                            n_points_x=None, mid_point=None):
    # The idea is that..
    # We create a plane of coordinates
    # With the interpretation of...
    # (fh, ap, rl)
    # Which we also define as
    # (z, y, x)
    # This means that this order is also constrained on the max, offset and angle lists and mid-point
    max_z, max_y, max_x = max_list
    delta_z, delta_y, delta_x = delta_list
    angle_z, angle_y, angle_x = angle_list

    if n_points_y is None:
        n_points_y = max_y + delta_y
    if n_points_x is None:
        n_points_x = max_x + delta_x
    if mid_point is None:
        mid_point = np.zeros(3)

    # Make sure that the z-coordinate is always zero. That is where we start
    # That is where we rotate
    mid_point[0] = 0
    if isinstance(mid_point, list):
        mid_point = np.array(mid_point)

    if isinstance(offset_list, list):
        offset_list = np.array(offset_list)

    y_range = np.linspace(0 - delta_y, max_y + delta_y, n_points_y)
    x_range = np.linspace(0 - delta_x, max_x + delta_x, n_points_x)

    Y, X = np.meshgrid(y_range, x_range, indexing='ij')
    # Here is the reason why we take the first cordinate of mid_point to be zero
    Z = np.zeros(Y.shape)  # I believe this is so because we only take on slice..

    slice_coords = np.stack([Z, Y, X])
    R_z = harray.rot_z(angle_z)
    R_y = harray.rot_y(angle_y)
    R_x = harray.rot_x(angle_x)
    # This order of rotation should now be the same as with the order method..
    R_total = R_x @ R_y @ R_z
    # Reverse this since we apply the rotation in a different order..?
    print('Rotation matrix \n', R_total)
    # Rotate the whole meshgrid we just created
    acq_plane_rot = []
    for ix, iy, iz in zip(slice_coords[0].ravel(), slice_coords[1].ravel(), slice_coords[2].ravel()):
        temp = [ix, iy, iz]
        rot_temp = (R_total @ (temp - mid_point)) + mid_point
        acq_plane_rot.append(rot_temp)

    acq_plane_rot = np.array(acq_plane_rot) + np.array(offset_list).reshape(1, 3)

    return acq_plane_rot


def get_file_content(file_name):
    with h5py.File(file_name, 'r') as f:
        img_struct = f['ProcessedData']
        sigma_array = np.array(img_struct['sigma'])
        rho_array = np.array(img_struct['rho'])
        mask_array = np.array(img_struct['Bodymask'])
        # The first index contains the b1 plus
        b1_plus_array = img_struct['B1'][:, 0]
        b1_plus_array_cpx = b1_plus_array['real'] + 1j * b1_plus_array['imag']

        # The second one the b1 minus
        b1_minus_array = img_struct['B1'][:, 1]
        b1_minus_array_cpx = b1_minus_array['real'] + 1j * b1_minus_array['imag']

    # if 'ella' in file_name.lower():
    #     sigma_array = np.rot90(sigma_array, axes=(-2, -1))
    #     b1_plus_array_cpx = np.rot90(b1p_array, axes=(-2, -1))
    #     b1_minus_array_cpx = np.rot90(b1m_array, axes=(-2, -1))
    #     mask_array = np.rot90(mask_array, axes=(-2, -1))

    return {'sigma': sigma_array, 'b1p': b1_plus_array_cpx, 'rho': rho_array,
            'b1m': b1_minus_array_cpx, 'body_mask': mask_array}

"""
Create destination dir...
"""

data_b1_dir = '/media/bugger/MyBook/data/simulated/cardiac/bart'
hmisc.create_datagen_dir(data_b1_dir, type_list=['sa'], data_list=['b1_minus', 'b1_plus', 'rho', 'mask', 'sigma'])

sigma_dir = os.path.join(data_b1_dir, 'sa', 'sigma')
rho_dir = os.path.join(data_b1_dir, 'sa', 'rho')
b1min_dir = os.path.join(data_b1_dir, 'sa', 'b1_minus')
b1plus_dir = os.path.join(data_b1_dir, 'sa', 'b1_plus')
mask_dir = os.path.join(data_b1_dir, 'sa', 'mask')


"""
Data transformation parameters
"""

# These parameters process the data such that we have SA views on the heart...
operator_dict = {"operator_Ella": {'angle1': 30, 'angle2': -30, 'angle3': -50, 'slice_min': 60, 'slice_max': 65},
"operator_V10" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50},
"operator_V11" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50},
"operator_V12" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50},
"operator_V13" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50},
"operator_V14" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50},
"operator_V1" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50},
"operator_V2" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50},
"operator_V3" : {'angle1': 40, 'angle2': 20, 'angle3': 0,'slice_min': 45, 'slice_max': 50},
"operator_V4" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50},
"operator_V5" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50},
"operator_V6" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50},
"operator_V7" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50},
"operator_V8" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50},
"operator_V9" : {'angle1': 40, 'angle2': 20, 'angle3': 0, 'slice_min': 45, 'slice_max': 50}}

"""
Loop over files..
"""

import time
cardiac_b1_files = [os.path.join(data_b1_dir, x) for x in os.listdir(data_b1_dir) if x.endswith('mat')]
for sel_cardiac_b1_file in cardiac_b1_files[1:]:
    t0 = time.time()
    re_obj = re.findall("(\w+)_ProcessedData", sel_cardiac_b1_file)[0]
    operator_name = f'operator_{re_obj}'
    operator_parameters = operator_dict[operator_name]

    print('Starting with ', operator_name)
    container_dict = get_file_content(sel_cardiac_b1_file)
    sigma_array = container_dict['sigma']
    rho_array = container_dict['rho']
    b1p_array = container_dict['b1p']
    b1m_array = container_dict['b1m']
    body_mask_array = container_dict['body_mask']
    # hplotc.SlidingPlot(sigma_array)

    AngulatedSlice(sigma_array)
    # Im going for only one slice to save some time....
    avg_slice = int((operator_parameters['slice_max'] + operator_parameters['slice_min']) / 2)
    operator_parameters['slice_min'] = avg_slice
    operator_parameters['slice_max'] = avg_slice + 1

    angle_list = [operator_parameters['angle1'], operator_parameters['angle2'], operator_parameters['angle3']]
    offset_list = [avg_slice, 0, 0]
    voxel_factor = 2
    npoints_x = int(sigma_array.shape[2]*voxel_factor)
    npoints_y = int(sigma_array.shape[1]*voxel_factor)
    mid_point = np.array(sigma_array.shape)//2
    mid_point[0] = 0
    # Here we obtained the plane of coordinates that we will use to slice each array
    rotated_plane_coords = get_affine_slice_coords(max_list=sigma_array.shape,
                                                 delta_list=[0, 0, 0],
                                                 angle_list=angle_list,
                                                 offset_list=offset_list,
                                                 n_points_x=npoints_x,
                                                 n_points_y=npoints_y,
                                                 mid_point=mid_point)

    # Here we apply this array to the sigma array
    mapped_sigma_array = scipy.ndimage.map_coordinates(sigma_array, rotated_plane_coords.T, cval=-1, order=3).reshape((npoints_y, npoints_x))
    # Add an axis.. used for later..
    mapped_sigma_array = mapped_sigma_array[None]

    # Here we apply this array to the sigma array
    mapped_rho_array = scipy.ndimage.map_coordinates(rho_array, rotated_plane_coords.T, cval=-1, order=3).reshape(
        (npoints_y, npoints_x))

    #
    # Here we get the sliced body mask
    mapped_body_mask_array = scipy.ndimage.map_coordinates(body_mask_array, rotated_plane_coords.T, cval=0, order=3).reshape(
        (npoints_y, npoints_x))
    mapped_body_mask_array = harray.get_treshold_label_mask(mapped_body_mask_array)
    # Add an axis.. used for later..
    mapped_body_mask_array = mapped_body_mask_array[None]

    # Here we get the B1 plus and B1 minus maps
    mapped_b1p_array = np.stack([scipy.ndimage.map_coordinates(x, rotated_plane_coords.T, cval=0, order=3).reshape((npoints_y, npoints_x)) for x in b1p_array], axis=0)
    mapped_b1p_array = mapped_b1p_array[None]
    mapped_b1m_array = np.stack([scipy.ndimage.map_coordinates(x, rotated_plane_coords.T, cval=0, order=3).reshape((npoints_y, npoints_x)) for x in b1m_array], axis=0)
    mapped_b1m_array = mapped_b1m_array[None]

    # Here we crop the sigma array, with the body mask
    crop_sigma_array, crop_body_mask = zip(*[harray.get_crop(x, x_mask=y, silent=True) for x, y in zip(mapped_sigma_array, mapped_body_mask_array)])

    # Also crop the rho-array
    crop_rho_array, _ = harray.get_crop(mapped_rho_array, x_mask=mapped_body_mask_array[0], silent=True)
    crop_rho_array = crop_rho_array[None]
    # Here we get cropped b1p versions
    temp = [[harray.get_crop(x, x_mask=y, silent=True) for x in x_list] for x_list, y in zip(mapped_b1p_array, mapped_body_mask_array)]
    crop_b1p_array = np.array(temp)[:, :, 0]

    # Here we get cropped b1m versions
    temp = [[harray.get_crop(x, x_mask=y, silent=True) for x in x_list] for x_list, y in zip(mapped_b1m_array, mapped_body_mask_array)]
    crop_b1m_array = np.array(temp)[:, :, 0]

    # Now store stuff...... as Numpy, that is what the registration process expects...
    n_loc = mapped_sigma_array.shape[0]
    for i_loc in range(n_loc):
        np.save(os.path.join(sigma_dir, re_obj), crop_sigma_array[i_loc])
        np.save(os.path.join(rho_dir, re_obj), crop_rho_array[i_loc])
        np.save(os.path.join(mask_dir, re_obj), crop_body_mask[i_loc])
        np.save(os.path.join(b1plus_dir, re_obj), crop_b1p_array[i_loc])
        np.save(os.path.join(b1min_dir, re_obj), crop_b1m_array[i_loc])

    print('Time', time.time() - t0)
