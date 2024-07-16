import scipy.ndimage.interpolation as scint
import numpy as np

import numpy as np
import sys

import PyQt5.QtGui
import PyQt5.QtCore
import PyQt5.QtWidgets

import matplotlib.pyplot as plt
import helper.misc as hmisc
import scipy.optimize
import PyQt5.QtGui
import PyQt5.QtCore
import PyQt5.QtWidgets
from matplotlib.backend_bases import MouseButton

from tooling.genericinterface import GenericInterface
import scipy.ndimage
import os
import re


def change_list_order(list_format):
    # Change from [['a', 'b', 'c'], ['a', 'b', 'c']]
    # to [['a', 'a'], ['b', 'b'], ['c', 'c']]
    # and vica versa
    new_format = [[] for _ in range(len(list_format[0]))]
    for i in list_format:
        for ii, j in enumerate(i):
            new_format[ii].append(j)
            # print(f'{ii}, {j} \t {new_format}')
    return new_format


"""
Tool created to slice data...
"""


class SliceToolArray(GenericInterface):
    def __init__(self, raw_image, max_ds=10, nrows_plot=1, ncols_plot=2, **kwargs):
        super().__init__()

        self.debug = kwargs.get('debug', False)
        # Only needed to write the output of apply_button
        self.dest_dir = kwargs.get('dest_dir', '')
        self.max_ds = max_ds
        self.nrows_plot = nrows_plot
        self.ncols_plot = ncols_plot

        self.slider_list = None  # We need to define this ...
        # Define the canvas
        self.canvas, self.axes = self.get_canvas(nrows=self.nrows_plot, ncols=self.ncols_plot)

        if hasattr(self.axes, 'ravel'):
            self.axes = self.axes.ravel()
        # Check de files in een specifieke dest_dir...

        # Initializatirs for variables we are going to use
        self.reset_properties()

        self.raw_image_image = raw_image
        image = self.proc_image(raw_image)
        self.img_shape_list[0] = image.shape
        self.img_array_list[0] = image
        self.plot_array_list[0] = image[self.sel_slice_list[0]]

        # ============== PyQt5 Impl ==============
        h_layout = PyQt5.QtWidgets.QHBoxLayout(self._main)

        # Add buttons to the box layout
        self.button_box_layout = self.get_button_box_layout()
        # Calculates a function based off the polygon that is draw and the left array
        calc_button = self.get_push_button(name='Calculate', shortcut='C', connect=self.calculate)
        # Toggles visibility of polygon that has been created
        vis_button = self.get_push_button(name='Toggle visibility', shortcut='T', connect=self.set_vis)
        apply_button = self.get_push_button(name='Apply to all coils', connect=self.apply_button)

        self.button_box_layout.addWidget(calc_button)
        self.button_box_layout.addWidget(vis_button)
        self.button_box_layout.addWidget(apply_button)

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

    def proc_image(self, x):
        return np.abs(x[:, 0].sum(axis=0))

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
        self.print_status_objects()

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

    def apply_button(self):
        # Apply transformation to all coil images.. and select the slices
        n_c, n_stack = self.raw_image_image.shape[:2]
        print('Rotation amount of coils: ', n_c)

        for i_stack in range(n_stack):
            print('Stack ', i_stack)
            print('\t Applying rotation for coil...')
            result_rot = []
            for i in range(n_c):
                print('\t', i)
                i_img = self.raw_image_image[i, i_stack]
                print('\t Shape ', i_img.shape)
                # Get the angle of the first thing only.. (hard coded)
                sel_degree = self.angle_plot_list[0]
                sel_slice = self.sel_slice_list[0]
                temp_img_array_rot = self.apply_rotation(i_img, temp_ax=0, degree=sel_degree)
                sel_array_rot = temp_img_array_rot[sel_slice]
                result_rot.append(sel_array_rot)

            result_rot_np = np.array(result_rot)
            print('Finished ', result_rot_np.shape)
            np.save(os.path.join(self.dest_dir, f'applied_rot_{i_stack}'), result_rot_np)

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

            print('\t Line xy data shape ', temp_data.shape)
            print('\t Line xy data ', temp_data)

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

            print('Resulting image size ', temp_img_array.shape)

            print('Applying rotation...')
            temp_img_array_rot = self.apply_rotation(temp_img_array, temp_ax=self.temp_ax, degree=a_degree)

            print('Resulting image size ', temp_img_array_rot.shape)

            sel_slice = int(x_coord_new)

            print('Selecting new image...')
            sel_array_rot = temp_img_array_rot[sel_slice]

            self.img_array_list[self.temp_ax + 1] = temp_img_array_rot
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
            self.axes_imshow[self.temp_ax+1] = None
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
    # Simple test case...
    # A_term = np.moveaxis(np.squeeze(A), -1, 2)
    A_term = np.random.rand(1, 1, 10, 10, 10)

    qapp = PyQt5.QtWidgets.QApplication(sys.argv)
    app = SliceToolArray(A_term, ncols_plot=3, dest_dir='/home/bugger')
    app.show()
    qapp.exec_()
