import scipy.ndimage.interpolation as scint
import numpy as np

import numpy as np
import sys

import PyQt5.QtGui
import PyQt5.QtCore
import PyQt5.QtWidgets

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.backend_bases import MouseButton

import helper.misc as hmisc
import helper.array_transf as harray
import scipy.optimize
import PyQt5.QtGui
import PyQt5.QtCore
import PyQt5.QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.backend_bases import MouseButton

from tooling.genericinterface import GenericInterface
import scipy.ndimage
import helper.plot_fun as hplotf
import os
import re


"""
Tool created to slice data...
"""


class SliceTool(GenericInterface):
    # We hope to create this tool such that we have a generic Mask-interaciton tool
    # Child classes of this will have different effect on WHAT we do with the data

    def __init__(self, data_dir, dest_dir, max_ds=10, nrows_plot=1, ncols_plot=2, file_ext='mat', **kwargs):
        super().__init__()

        self.debug = kwargs.get('debug', False)

        self.max_ds = max_ds
        self.nrows_plot = nrows_plot
        self.ncols_plot = ncols_plot

        self.data_dir = data_dir
        self.dest_dir = dest_dir
        self.dest_dir_type_list = ['p2ch', 'p4ch', 'axial']
        self.dest_dir_data_list = ['b1_minus', 'b1_plus', 'rho']

        self.slider_list = None  # We need to define this ...
        # Define the canvas
        self.canvas, self.axes = self.get_canvas(nrows=self.nrows_plot, ncols=self.ncols_plot)

        if hasattr(self.axes, 'ravel'):
            self.axes = self.axes.ravel()
        # Check de files in een specifieke dest_dir...

        self.file_ext = file_ext
        list_files = os.listdir(self.data_dir)
        self.list_files = [x for x in list_files if x.endswith(file_ext)]

        self.view_id = 0
        self.file_name = self.list_files[self.view_id]
        img_struct = self.read_struct(self.data_dir, self.file_name)

        self.reset_properties()
        self.set_new_file(img_struct)

        # ============== PyQt5 Impl ==============
        h_layout = PyQt5.QtWidgets.QHBoxLayout(self._main)

        # Add buttons to the box layout
        self.button_box_layout = self.get_button_box_layout()
        # Calculates a function based off the polygon that is draw and the left array
        calc_button = self.get_push_button(name='Calculate', shortcut='C', connect=self.calculate)
        # Toggles visibility of polygon that has been created
        vis_button = self.get_push_button(name='Toggle visibility', shortcut='T', connect=self.set_vis)
        # Shows the images on the B1 minus and plus maps
        preview_button = self.get_push_button(name='Preview slices', shortcut='L', connect=self.get_preview)
        # Writes the B1 plus and B1 minus images... also the rho ones..?
        # Yeah lets do that
        write_button = self.get_push_button(name='Write', shortcut='W', connect=self.write_b1)

        self.button_box_layout.addWidget(calc_button)
        self.button_box_layout.addWidget(vis_button)
        self.button_box_layout.addWidget(preview_button)
        self.button_box_layout.addWidget(write_button)

        # Callback definition
        self.canvas.mpl_connect('draw_event', self.draw_callback)
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)


        self.layout_list, self.slider_list = zip(*[self.get_slider_canvas(ax_index=x) for x in range(ncols_plot)])
        temp_master_slider = PyQt5.QtWidgets.QHBoxLayout()
        for i_slider in self.layout_list:
            temp_master_slider.addLayout(i_slider)

        temp_canvas_layout = PyQt5.QtWidgets.QVBoxLayout()
        temp_canvas_layout.addWidget(self.canvas, 7)
        temp_canvas_layout.addLayout(temp_master_slider, 1)

        h_layout.addLayout(temp_canvas_layout)
        h_layout.addLayout(self.button_box_layout)

        self.update_plots()
        self.canvas.draw()

    def open_file(self):
        name, _ = PyQt5.QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
                                                              self.data_dir, "",
                                                              options=PyQt5.QtWidgets.QFileDialog.DontUseNativeDialog)

        if name:  # Check if we got anything
            print('Opened file ', name)
            img_struct = self.read_struct(os.path.dirname(name), os.path.basename(name))
            self.reset_properties()
            self.set_new_file(img_struct)

    def write_b1(self):
        print('Writing')

        for i_type in self.dest_dir_type_list:
            for i_data in self.dest_dir_data_list:
                temp_file_name = re.sub(' ', '_', self.file_name)
                temp_file_name, ext = os.path.splitext(temp_file_name)
                # We are going to write away numpy files
                temp_path = os.path.join(self.dest_dir, i_type, i_data, temp_file_name + '.npy')
                print('\t Writing towards', temp_path)

                if i_data == 'b1_minus':
                    temp_list = self.b1_minus_array_list
                elif i_data == 'b1_plus':
                    temp_list = self.b1_plus_array_list
                elif i_data == 'rho':
                    temp_list = self.img_array_list
                else:
                    print('Eerrrr')

                # ['p2ch', 'p4ch', 'axial']
                if i_type == 'axial':
                    temp_slice = self.sel_slice_list[0]
                    temp_array = temp_list[0]
                elif i_type == 'p2ch':
                    temp_slice = self.sel_slice_list[1]
                    temp_array = temp_list[1]
                elif i_type == 'p4ch':
                    temp_slice = self.sel_slice_list[2]
                    temp_array = temp_list[2]
                else:
                    temp_array = None
                    print('Errrr')

                if temp_array is not None:
                    # Added extra dimension to make sure it is in line with coil dimension
                    print('temp array shape ', temp_array.shape)
                    if temp_array.ndim == 3:
                        temp_array = temp_array[np.newaxis]

                    print('temp array shape ', temp_array.shape)
                    print('temp array shape ', temp_array[:, temp_slice].shape)

                    np.save(temp_path, temp_array[:, temp_slice])
                else:
                    print(' Array is none....')

    def read_struct(self, path, i_file):
        print('Reading struct')
        file_dir = os.path.join(path, i_file)
        print('\t', file_dir)
        h5_struct = None
        if file_dir.endswith(self.file_ext):
            # Now we properly close this thing as well
            with h5py.File(file_dir, 'r') as h5_file:
                temp_rho = np.array(h5_file['ProcessedData']['rho'])
                temp_b1 = np.array(h5_file['ProcessedData']['B1'])

        array_dict = {'rho': temp_rho, 'B1': temp_b1}
        return array_dict

    def reset_properties(self):
        # Get the current id that we are viewing from the list of iles

        [x.clear() for x in self.axes]
        # Ax definition // plot data
        self.temp_ind = None
        self.temp_data = None
        self.temp_ax = 0

        # Indicitors if we are moving or pressing
        self.press = False
        self.move = False
        self.line_vis = True

        # Initializatirs for variables we are going to use
        self.line_list = [None] * self.ncols_plot # * self.nrows_plot
        self.angle_plot_list = [None] * self.ncols_plot # * self.nrows_plot
        if self.slider_list is not None:
            [x.setValue(0) for x in self.slider_list]
        self.sel_slice_list = [0] * self.ncols_plot # * self.nrows_plot
        self.axes_imshow = [None] * self.ncols_plot # * self.nrows_plot
        self.img_shape_list = [None] * self.ncols_plot  # * self.nrows_plot

        self.img_array_list = [None] * self.ncols_plot  # * self.nrows_plot
        self.plot_array_list = [None] * self.ncols_plot  # * self.nrows_plot

        self.b1_plus_array_list = [None] * self.ncols_plot  # * self.nrows_plot
        self.b1_minus_array_list = [None] * self.ncols_plot  # * self.nrows_plot

    def set_new_file(self, img_struct):
        self.img_struct = img_struct
        self.img_array = self.img_struct['rho']

        # The first index contains the b1 plus
        self.b1_plus_array = self.img_struct['B1'][:, 0]
        self.b1_plus_array_list[0] = self.b1_plus_array['real'] + 1j * self.b1_plus_array['imag']

        # The second one the b1 minus
        self.b1_minus_array = self.img_struct['B1'][:, 1]
        self.b1_minus_array_list[0] = self.b1_minus_array['real'] + 1j * self.b1_minus_array['imag']

        self.img_shape_list[0] = self.img_array.shape
        self.img_array_list[0] = self.img_array
        self.plot_array_list[0] = self.img_array[self.sel_slice_list[0]]

        self.update_plots()
        self.canvas.draw()

    def change_file(self, mod=0):
        print('Change file')
        self.view_id += mod

        dest_dir = os.path.join(self.dest_dir, self.dest_dir_type_list[0], self.dest_dir_data_list[0])
        dest_files = os.listdir(dest_dir)
        print('We already have the following files read in..')
        [print('\t', x) for x in self.list_files]

        print('We already have the following files in destintion dir')
        [print('\t', x) for x in dest_files]
        print('\t In directory', dest_dir)

        # sel_file_list = [x for x in self.list_files if os.path.splitext(re.sub(' ', '_', x))[0] not in dest_files]
        sel_file_list = []
        for x in self.list_files:
            temp_file_name = os.path.splitext(re.sub(' ', '_', x))[0] + '.npy'
            bool_file = temp_file_name in dest_files
            if not bool_file:
                sel_file_list.append(x)

        print('Selection from the following files')
        [print('\t', x) for x in sel_file_list]

        n_files = len(sel_file_list)
        if n_files:
            self.view_id = self.view_id % n_files  # Make sure that we can rotate through
            self.file_name = sel_file_list[self.view_id]
            print('Changing to file ', self.file_name)
            img_struct = self.read_struct(self.data_dir, self.file_name)
        else:
            print('We have no more files left...')
            img_struct = None

        self.reset_properties()
        self.set_new_file(img_struct)

    def get_slider_canvas(self, ax_index):
        # self.temp_ax = ax_index

        temp_depth = self.img_shape_list[ax_index]
        if temp_depth is None:
            temp_depth = 0
        else:
            temp_depth = temp_depth[0]

        temp_lambda = lambda x: self.slider_change(x, ax_index=ax_index)
        temp_slider = self.get_slider(min=0, max=temp_depth, orientation='horizontal', connect=temp_lambda)

        temp_lambda = lambda x: self.slider_change(x, ax_index=ax_index, mod=1, slider_obj=temp_slider)
        button2right = self.get_push_button(connect=temp_lambda)

        temp_lambda = lambda x: self.slider_change(x, ax_index=ax_index, mod=-1, slider_obj=temp_slider)
        button2left = self.get_push_button(connect=temp_lambda)

        slider_layout = PyQt5.QtWidgets.QHBoxLayout()
        slider_layout.addWidget(button2left, 2)
        slider_layout.addWidget(temp_slider, 8)
        slider_layout.addWidget(button2right, 2)

        return slider_layout, temp_slider

    def slider_change(self, event, ax_index, mod=0, slider_obj=None):
        print('Slider change active. Using ax index ', ax_index)
        if slider_obj is not None:
            event = slider_obj.value()
        print(event)
        new_slice = max(event+mod, 0)
        print(self.img_array_list[ax_index].shape)
        if self.img_array_list[ax_index] is not None:

            print('\tSelecting image shape ', self.img_array_list[ax_index].shape)
            max_slice = self.img_array_list[ax_index].shape[0] - 1
            new_slice = min(max_slice, new_slice)
            self.plot_array_list[ax_index] = self.img_array_list[ax_index][new_slice]
            self.sel_slice_list[ax_index] = new_slice
            print('\tNew slice ', new_slice, ' / ', max_slice)
            print('\tPrint shape of plot array')
            print('\t\t', [x.shape for x in self.plot_array_list if x is not None])

        if slider_obj is not None:
            slider_obj.setValue(new_slice)

        self.update_plots()
        self.canvas.draw()

        print('/End slider change')

    def update_plots(self):
        # This function is separated from the call drawback on purpose
        # Updating the plots was a hell otherwise..

        # self.background = self.canvas.copy_from_bbox(self.left_ax.bbox)
        # temp_plot = [self.temp_ax]
        # Make sure that we have changed something to poly..
        # self.print_status_objects()

        print('Update plots')
        for i, temp_plot in enumerate(self.plot_array_list):
            if temp_plot is not None:
                print('\t Got data ', i, temp_plot.shape, temp_plot.mean())
                sel_axes_imshow = self.axes_imshow[i]
                if sel_axes_imshow is not None:
                    print('\t Imshow on ', i)
                    # print(sel_axes_imshow)
                    sel_axes_imshow.set_data(temp_plot)
                    vmin = temp_plot.min()
                    vmax = temp_plot.max()
                    sel_axes_imshow.set_clim(vmin=vmin, vmax=vmax)
                    # I am not sure if everything is stored back properly
                    self.axes_imshow[i] = sel_axes_imshow
                else:
                    print('\t Setting stuff on ', i)
                    self.axes_imshow[i] = self.axes[i].imshow(temp_plot)
                    # self.axes[i].imshow(temp_plot)
                #
                # if i == 1:
                #     print('Changing aspect ratio')
                #     self.axes[i].set_aspect(aspect=20)

    def draw_callback(self, event):
        print('Draw callback')
        for i, i_line in enumerate(self.line_list):
            if i_line is not None:
                print('\tDrawing line ', i_line)
                self.axes[i].add_line(i_line)
                self.axes[i].draw_artist(i_line)

        print('/End draw callback')

    def button_release_callback(self, event):
        if event.inaxes != self.axes[self.temp_ax]:
            print('Break down')
            return
        else:
            sel_line = self.line_list[self.temp_ax]
            print('\nSelected line ', sel_line)

            if sel_line is None and event.button == MouseButton.LEFT:
                sel_line = plt.Line2D([self.temp_data[0], event.xdata],
                                      [self.temp_data[1], event.ydata],
                                      color='black', marker='o', mfc='r', alpha=0.8, animated=True)
                self.axes[self.temp_ax].add_line(sel_line)
                self.line_list[self.temp_ax] = sel_line

        self.press = False
        self.moving = False

        self.canvas.draw()
        #self.update_plots()

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        pressed_ax = event.inaxes

        if pressed_ax not in self.axes:
            return

        self.temp_ax = self.axes.tolist().index(pressed_ax)

        temp_line = self.line_list[self.temp_ax]

        if temp_line is None:
            self.temp_data = [event.xdata, event.ydata]
            print('New line!')
            print('\t Using data ', self.temp_data)
        else:
            self.temp_ind = self.get_ind_under_cursor(event)

            if self.temp_ind is None:
                print('Remove...line?')
                self.remove_line()
                self.temp_data = [event.xdata, event.ydata]

        # Notify that we are pressing something
        self.press = True

        # If we have pressed the right button.. then remove the line
        if event.button == MouseButton.RIGHT:
            self.remove_line()

    def motion_notify_callback(self, event):
        "on mouse movement"

        ignore = (event.inaxes is None or event.button != 1 or self.temp_ind is None)

        # Only passes through when we are draggin a node.
        if ignore:
            return

        x, y = event.xdata, event.ydata

        if self.press:
            if not self.moving:
                self.temp_ind = self.get_ind_under_cursor(event)

            print('Pressed and moving ', self.temp_ind)
            self.moving = True

            self.temp_ax = self.axes.tolist().index(event.inaxes)
            temp_line = self.line_list[self.temp_ax]
            temp_xy = temp_line.get_xydata()
            temp_xy[self.temp_ind, :] = [x, y]
            #
            temp_line.set_data(temp_xy.T)
            self.line_list[self.temp_ax] = temp_line
            # self.update_plots()
            self.canvas.draw()

    def calculate(self):
        print('Calculate called')
        # Get the index of the last image..
        # Not fool proof, but ok.
        last_line_index = self.line_list.index(None) - 1
        # We cant have it to be negative...
        if last_line_index >= 0:
            print('\tSetting ax index to ', last_line_index)
            print('\t However.. we also have this index..', self.temp_ax)
            temp_line = self.line_list[self.temp_ax]
            # self.temp_ax = last_line_index
            temp_depth, temp_ny, temp_nx = self.img_shape_list[self.temp_ax]

            print('\t Obtained the following image parametrs ', temp_depth, temp_ny, temp_nx)
            temp_data = temp_line.get_xydata()
            xcoord = temp_data[:, 0]
            ycoord = temp_data[:, 1]

            sort_id = np.argsort(xcoord)
            xcoord = np.sort(xcoord)
            ycoord = ycoord[sort_id]
            a = np.diff(ycoord)/np.diff(xcoord)
            b = ycoord[1] - a * xcoord[1]

            a_degree = np.arctan(-1/a[0]) / np.pi * 180
            self.angle_plot_list[self.temp_ax] = a_degree
            print('\tCalculated offset', b)
            print('\tCalculated degree', a_degree)

            coord_x_1 = (0 - b) / a
            coord_x_2 = (temp_ny - b) / a
            coord_y_1 = temp_nx * a + b
            coord_y_2 = b

            coords = []
            if 0 < coord_x_1 < temp_nx:
                coords.append((float(coord_x_1), 0))
            if 0 < coord_x_2 < temp_nx:
                coords.append((float(coord_x_2), temp_ny - 1))
            if 0 < coord_y_1 < temp_ny:
                coords.append((temp_nx - 1, float(coord_y_1)))
            if 0 < coord_y_2 < temp_ny:
                coords.append((0, float(coord_y_2)))

            coord_x, coord_y = hmisc.change_list_order(coords)
            sort_id = np.argsort(coord_x)
            sort_x = np.sort(coord_x)
            sort_y = np.array(coord_y)[sort_id]

            # Calculate the position in the newly rotated frame pased off of these
            line_xy_data = self.line_list[self.temp_ax].get_xydata()
            print('\t Line xy data shape ', line_xy_data.shape)
            print('\t Line xy data ', line_xy_data)

            bool_sort_x = sorted(sort_x) == [0, temp_nx-1]
            bool_sort_y = sorted(sort_y) == [0, temp_ny - 1]
            print('check sort_x', bool_sort_x)
            print('check sort_y', bool_sort_y)

            if bool_sort_y:
                x_term_1 = temp_ny * np.sin(np.radians(a_degree))
                x_term_2 = sort_x[0] * np.cos(np.radians(a_degree))
                x_coord_new = np.abs(x_term_1) + np.abs(x_term_2)

                y_term_1 = temp_ny * np.cos(np.radians(a_degree))
                y_term_2 = temp_nx * np.sin(np.radians(a_degree))

                y_coord_new = np.abs(y_term_1) + np.abs(y_term_2)

            if bool_sort_x:
                x_term_1 = sort_y[0] * np.sin(np.radians(a_degree))
                x_term_2 = 0
                x_coord_new = np.abs(x_term_1)

                y_term_1 = temp_ny * np.cos(np.radians(a_degree))
                y_term_2 = temp_nx * np.sin(np.radians(a_degree))

                y_coord_new = np.abs(y_term_1) + np.abs(y_term_2)

            print('\t Output of sort_x ', sort_x)
            print('\t Output of sort_y ', sort_y)

            print('\t x- first term ', x_term_1)
            print('\t x- second term ', x_term_2)
            print('\t\t x- combined ', np.abs(x_term_1) + np.abs(x_term_2))

            print('\t y- first term ', y_term_1)
            print('\t y- second term ', y_term_2)
            print('\t\t y- combined ', np.abs(y_term_1) + np.abs(y_term_2))

            # Perform all the transformations on the B1plus, B1minus and Rho data...
            # Get the image from the current axis (being the one with the lastly drawed line)
            temp_img_array = np.array(self.img_array_list[self.temp_ax])
            temp_b1_plus_array = np.array(self.b1_plus_array_list[self.temp_ax])
            temp_b1_minus_array = np.array(self.b1_minus_array_list[self.temp_ax])

            print('Resulting image size ', temp_img_array.shape)
            print('Resulting image size ', temp_b1_minus_array.shape)

            print('Applying rotation...')
            temp_img_array_rot = self.apply_rotation(temp_img_array, temp_ax=self.temp_ax, degree=a_degree)
            temp_b1_plus_array_rot = self.apply_rotation(temp_b1_plus_array, temp_ax=self.temp_ax, degree=a_degree)
            temp_b1_minus_array_rot = self.apply_rotation(temp_b1_minus_array, temp_ax=self.temp_ax, degree=a_degree)

            print('Resulting image size ', temp_img_array_rot.shape)
            print('Resulting image size ', temp_b1_minus_array_rot.shape)

            sel_slice = int(x_coord_new)
            print('Selecting new image...')
            sel_array_rot = temp_img_array_rot[sel_slice]

            self.img_array_list[self.temp_ax + 1] = temp_img_array_rot
            self.b1_plus_array_list[self.temp_ax + 1] = temp_b1_plus_array_rot
            self.b1_minus_array_list[self.temp_ax + 1] = temp_b1_minus_array_rot

            self.img_shape_list[self.temp_ax + 1] = temp_img_array_rot.shape
            # Set the new slice
            self.sel_slice_list[self.temp_ax + 1] = sel_slice
            # Set the new plot array
            self.plot_array_list[self.temp_ax + 1] = sel_array_rot
            # Set the new slider value
            new_max_depth = temp_img_array_rot.shape[0]
            self.slider_list[self.temp_ax + 1].setMaximum(new_max_depth)
            self.slider_list[self.temp_ax + 1].setValue(self.sel_slice_list[self.temp_ax+1])

            if self.temp_ax == 1:
                pass
                # self.axes[self.temp_ax].scatter(sort_x[0], sort_y[0])
                # self.axes[self.temp_ax].scatter(sort_x[1], sort_y[1])
                # Used to double check the plane in the rotated image
                # self.axes[self.temp_ax + 1].vlines(x_coord_new, 0, y_coord_new, 'k')

            print('/Done calculating')
            self.update_plots()
            self.canvas.draw()

    def apply_rotation(self, temp_array, temp_ax, degree):
        # Rotate this an a_degree
        print('Applying rotation.. shape ', temp_array.shape)
        if 'complex' in str(temp_array.dtype):
            # Apply transformations seperately to each channel
            if temp_array.ndim == 4:
                temp_array_rot_real = scipy.ndimage.rotate(temp_array.real, angle=degree, axes=(3, 2))
                temp_array_rot_imag = scipy.ndimage.rotate(temp_array.imag, angle=degree, axes=(3, 2))
                temp_array_rot = temp_array_rot_real + 1j * temp_array_rot_imag
            elif temp_array.ndim == 3:
                temp_array_rot_real = scipy.ndimage.rotate(temp_array.real, angle=degree, axes=(2, 1))
                temp_array_rot_imag = scipy.ndimage.rotate(temp_array.imag, angle=degree, axes=(2, 1))
                temp_array_rot = temp_array_rot_real + 1j * temp_array_rot_imag
            else:
                temp_array_rot = None
        else:
            temp_array_rot = scipy.ndimage.rotate(temp_array, angle=degree, axes=(2, 1))

        print('Applying rotation.. shape ', temp_array.shape)
        # Set the new image as the rotated version
        if temp_ax == 0:
            if temp_array_rot.ndim == 4:
                temp_array_rot = np.moveaxis(temp_array_rot, (1, 2, 3), (2, 3, 1))[:, :, ::-1]
            elif temp_array_rot.ndim == 3:
                temp_array_rot = np.moveaxis(temp_array_rot, (0, 1, 2), (1, 2, 0))[:, ::-1]
            else:
                print('We have something weird in apply rotation..')
        else:
            if temp_array_rot.ndim == 4:
                temp_array_rot = np.moveaxis(temp_array_rot, (1, 2, 3), (3, 2, 1))[:, :, ::-1]
            elif temp_array_rot.ndim == 3:
                temp_array_rot = np.moveaxis(temp_array_rot, (0, 1, 2), (2, 1, 0))[:, ::-1]
            else:
                print('We have something weird in apply rotation..')

            temp_angle = self.angle_plot_list[0]
            print('Reverting this angle...', temp_angle)
            if 'complex' in str(temp_array.dtype):
                if temp_array_rot.ndim == 4:
                    temp_array_rot_real = scipy.ndimage.rotate(temp_array_rot.real, angle=-temp_angle, axes=(3, 2))
                    temp_array_rot_imag = scipy.ndimage.rotate(temp_array_rot.imag, angle=-temp_angle, axes=(3, 2))
                    temp_array_rot = temp_array_rot_real + 1j * temp_array_rot_imag
                elif temp_array_rot.ndim == 3:
                    temp_array_rot_real = scipy.ndimage.rotate(temp_array_rot.real, angle=-temp_angle, axes=(2, 1))
                    temp_array_rot_imag = scipy.ndimage.rotate(temp_array_rot.imag, angle=-temp_angle, axes=(2, 1))
                    temp_array_rot = temp_array_rot_real + 1j * temp_array_rot_imag
                else:
                    temp_array_rot = None
            else:
                temp_array_rot = scipy.ndimage.rotate(temp_array_rot, angle=-temp_angle, axes=(2, 1))

        return temp_array_rot

    def get_preview(self):
        print('Preview')
        for i in range(self.ncols_plot):
            temp_slice = self.sel_slice_list[i]
            print('\t Getting slice ', temp_slice)
            temp_array = self.b1_minus_array_list[i]
            if temp_array is not None:
                print('\t', i, temp_array.shape)
                print('\t Sel slice ', temp_slice)
                temp_plot = temp_array[:, temp_slice].sum(axis=0)
                self.axes[i + self.ncols_plot].imshow(np.abs(temp_plot))
        self.canvas.draw()

    def print_status_objects(self):
        print('Selected lines')
        [print(x) for x in self.line_list]
        print('Selected slices')
        [print(x) for x in self.sel_slice_list]
        print('Plot object shapes')
        [print(i, x.shape) for i, x in enumerate(self.plot_array_list) if x is not None]
        print('Img object shapes')
        [print(i, x.shape) for i, x in enumerate(self.img_array_list) if x is not None]
        print('Obtained angles')
        [print(x) for x in self.angle_plot_list]

    def get_ind_under_cursor(self, event):
        'get the index of the vertex under cursor if within max_ds tolerance'
        # display coords
        self.temp_ax = self.axes.tolist().index(event.inaxes)
        temp_line = self.line_list[self.temp_ax]
        if temp_line is not None:
            # Hier vergelijken we hm mee..
            xyt = np.asarray(temp_line._xy)
            xt, yt = xyt[:, 0], xyt[:, 1]
            d = np.sqrt((xt - event.xdata)**2 + (yt - event.ydata)**2)
            indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
            ind = indseq[0]
            if d[ind] >= self.max_ds:
                ind = None
        else:
            ind = None

        return ind

    def set_vis(self):
        for i_line in range(len(self.line_list)):
            if self.line_list[i_line] is not None:
                self.line_list[i_line].set_visible(not self.line_vis)
                self.canvas.draw()
                self.line_vis = not self.line_vis

    def remove_line(self):
        self.line_list[self.temp_ax] = None

        self.temp_data = []
        self.temp_ind = None


if __name__ == "__main__":

    import os
    import h5py

    # data_b1_minus_dir = '/home/bugger/Documents/data/simulation/cardiac/b1'
    data_b1_dir = '/media/bugger/MyBook/backup_files_folders/laptop_bugger_data/data/simulation/cardiac/b1'
    dest_dir = '/home/bugger/Documents/data/simulation/cardiac/b1'

    qapp = PyQt5.QtWidgets.QApplication(sys.argv)
    app = SliceTool(data_b1_dir, dest_dir, debug=True, init_canvas=True, init_plot=True, ncols_plot=3, nrows_plot=2)
    app.show()
    qapp.exec_()
