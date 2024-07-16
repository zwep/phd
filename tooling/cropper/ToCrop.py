import matplotlib
matplotlib.use('Agg')

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
lets make a tool that crops images to 4:3 format...

Manually press the center
And manually create the width


"""


class CropTool(GenericInterface):
    def __init__(self, data_dir, dest_dir, file_ext='jpeg', **kwargs):
        super().__init__()

        self.ncols_plot = 2
        self.nrows_plot = 1
        self.max_ds = 10

        self.debug = kwargs.get('debug', False)

        self.data_dir = data_dir
        self.dest_dir = dest_dir

        # Left is the original, Right image will be the cutout
        self.canvas, self.axes = self.get_canvas(nrows=self.nrows_plot, ncols=self.ncols_plot)  # Define the canvas

        if hasattr(self.axes, 'ravel'):
            self.axes = self.axes.ravel()

        # This initializes more properties
        self.line_mode = False
        self.reset_properties()

        # Check de files in een specifieke dest_dir...
        self.file_ext = file_ext
        list_files = os.listdir(self.data_dir)
        self.list_files = [x for x in list_files if x.endswith(file_ext) or x.endswith('jpg')]
        self.view_id = -1

        # ============== PyQt5 Impl ==============
        h_layout = PyQt5.QtWidgets.QHBoxLayout(self._main)

        # Add buttons to the box layout
        self.button_box_layout = self.get_button_box_layout()
        # Calculates a function based off the polygon that is draw and the left array
        calc_button = self.get_push_button(name='Calculate', shortcut='C', connect=self.calculate)
        # Toggles visibility of polygon that has been created
        vis_button = self.get_push_button(name='Toggle visibility', shortcut='T', connect=self.set_vis)
        # Yeah lets do that
        write_button = self.get_push_button(name='Write', shortcut='W', connect=self.write_file)

        self.button_box_layout.addWidget(calc_button)
        self.button_box_layout.addWidget(vis_button)
        self.button_box_layout.addWidget(write_button)

        # Callback definition
        self.canvas.mpl_connect('draw_event', self.draw_callback)
        self.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

        temp_canvas_layout = PyQt5.QtWidgets.QVBoxLayout()
        temp_canvas_layout.addWidget(self.canvas, 7)

        h_layout.addLayout(temp_canvas_layout)
        h_layout.addLayout(self.button_box_layout)

        self.canvas.draw()

    def open_file(self):
        name, _ = PyQt5.QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
                                                              self.data_dir, "",
                                                              options=PyQt5.QtWidgets.QFileDialog.DontUseNativeDialog)
        if name:  # Check if we got anything
            if os.path.isfile(name):
                self.load_file(name)

        self.update_plots()
        self.canvas.draw()

    def load_file(self, name):
        print('Opened file ', name)
        self.reset_properties()
        img_array = hmisc.load_array(name)
        print(img_array.shape)
        self.plot_array_list[0] = img_array[:, :, 0]
        self.img_array_list[0] = img_array
        nx, ny = self.plot_array_list[0].shape
        title_ac = f"ratio {nx / ny} -- {ny / nx}"
        self.axes[0].set_title(title_ac)
        self.file_name = hmisc.get_base_name(name) + '.jpg'

    def update_plots(self):
        # This function is separated from the call drawback on purpose
        # Updating the plots was a hell otherwise..
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

    def change_file(self, mod=0):
        self.view_id += mod
        self.load_file(os.path.join(self.data_dir, self.list_files[self.view_id]))

        self.update_plots()
        self.canvas.draw()

    def write_file(self, max_size_kb=200):
            from PIL import Image
            from io import BytesIO
            temp_image = Image.fromarray(self.img_array_list[1])
            # Apply crop to
            img_byte_array = BytesIO()
            temp_image.save(img_byte_array, format='JPEG')

            # Compress the image while checking the file size
            quality = 85  # initial quality value
            # while img_byte_array.tell() > max_size_kb * 1024 and quality > 0:
            #    img_byte_array = BytesIO()  # reset the byte array
            #    temp_image.save(img_byte_array, format='JPEG', subsampling=0, quality=quality)
            #    quality -= 5

            # Save the compressed image to the output path
            # with open(os.path.join(self.dest_dir, self.file_name), 'wb') as output_file:
            #    output_file.write(img_byte_array.getvalue())
            print(quality)
            temp_image.save(os.path.join(self.dest_dir, self.file_name), format='JPEG', subsampling=0, quality=quality)

    def calculate(self):
        if self.line_list[0]:
            temp_line = self.line_list[0]
            xyt = np.asarray(temp_line._xy)
            xt, yt = xyt[:, 0], xyt[:, 1]
        else:
            xt = [0, self.plot_array_list[0].shape[1]]

        if self.point_list[0]:
            temp_point = self.point_list[0]
            midx, midy = temp_point.properties()['offsets'].data[0]
        else:
            midy, midx = np.array(self.plot_array_list[0].shape) // 2

        print('shape img ', self.plot_array_list[0].shape)
        print('x0    x1')
        print(xt[0], xt[1])

        print('mid x    mid y')
        print(midx, midy)
        ny, nx = self.plot_array_list[0].shape
        midx = int(midx)
        midy = int(midy)
        dist_x = int(np.abs(xt[0] - xt[1]))
        dist_y = int(4 / 3 * dist_x)
        min_y = max(midy - dist_y // 2, 0)
        max_y = min(midy + dist_y // 2, ny)
        min_x = max(midx - dist_x // 2, 0)
        max_x = min(midx + dist_x // 2, nx)

        self.plot_array_list[1] = self.plot_array_list[0][min_y: max_y, min_x:max_x]
        self.img_array_list[1] = self.img_array_list[0][min_y: max_y, min_x:max_x]
        title_ac = f"ratio {(max_y - min_y) / (max_x - min_x)}"
        self.axes[1].set_title(title_ac)
        self.update_plots()
        self.canvas.draw()

    def draw_callback(self, event):
        print('Draw callback')
        for i, i_line in enumerate(self.line_list):
            if i_line is not None:
                print('\tDrawing line ', i_line)
                self.axes[i].add_line(i_line)
                self.axes[i].draw_artist(i_line)

        print('/End draw callback')

    def key_press_callback(self, event):
        # To create a line
        if event.key == "a":
            self.line_mode = True
            print('line mode is True')

        if event.key == "s":
            self.line_mode = False
            print('line mode is False')

        # To create a dot
        if event.key == 'd':
            self.line_mode = False

    def button_release_callback(self, event):
        if event.inaxes != self.axes[self.temp_ax]:
            print('Break down')
            return
        else:
            if self.line_mode:
                sel_line = self.line_list[self.temp_ax]
                print('\nSelected line ', sel_line)

                if sel_line is None and event.button == MouseButton.LEFT:
                    sel_line = plt.Line2D([self.temp_data[0], event.xdata],
                                          [self.temp_data[1], event.ydata],
                                          color='black', marker='o', mfc='r', alpha=0.8, animated=True)
                    self.axes[self.temp_ax].add_line(sel_line)
                    self.line_list[self.temp_ax] = sel_line
            else:
                if self.point_list[self.temp_ax] is not None:
                    self.point_list[self.temp_ax].remove()
                scatter_ax = self.axes[self.temp_ax].scatter(event.xdata, event.ydata)
                self.point_list[self.temp_ax] = scatter_ax

        self.press = False
        self.moving = False

        self.canvas.draw()
        self.update_plots()

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        pressed_ax = event.inaxes

        # Return of not in axes
        if pressed_ax not in self.axes:
            return

        # Check which axes we pressed
        self.temp_ax = self.axes.tolist().index(pressed_ax)

        if self.line_mode:
            # Get the list of lines in the current axes
            temp_line = self.line_list[self.temp_ax]

            # If it is none, lets create the first line
            if temp_line is None:
                self.temp_data = [event.xdata, event.ydata]
                print('New line!')
                print('\t Using data ', self.temp_data)
            # Else...
            else:
                self.temp_ind = self.get_ind_under_cursor(event)

                if self.temp_ind is None:
                    print('Remove...line?')
                    self.remove_line()
                    self.temp_data = [event.xdata, event.ydata]

            # If we have pressed the right button.. then remove the line
            if event.button == MouseButton.RIGHT:
                self.remove_line()
        else:  # Now we only want to have one point.... activate upon press?
            # self.temp_data = [event.xdata, event.ydata]
            if self.point_list[self.temp_ax] is not None:
                self.point_list[self.temp_ax].remove()
            scatter_ax = self.axes[self.temp_ax].scatter(event.xdata, event.ydata)
            self.point_list[self.temp_ax] = scatter_ax

        # Notify that we are pressing something
        self.press = True

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
            if self.line_mode:
                self.temp_ax = self.axes.tolist().index(event.inaxes)
                temp_line = self.line_list[self.temp_ax]
                temp_xy = temp_line.get_xydata()
                temp_xy[self.temp_ind, :] = [x, y]
                #
                temp_line.set_data(temp_xy.T)
                self.line_list[self.temp_ax] = temp_line
                self.canvas.draw()
            else:
                #self.temp_data = [event.xdata, event.ydata]
                if self.point_list[self.temp_ax] is not None:
                    self.point_list[self.temp_ax].remove()
                scatter_ax = self.axes[self.temp_ax].scatter(event.xdata, event.ydata)
                self.point_list[self.temp_ax] = scatter_ax

    def get_ind_under_cursor(self, event):
        'get the index of the vertex under cursor if within max_ds tolerance'
        # display coords
        self.temp_ax = self.axes.tolist().index(event.inaxes)
        temp_line = self.line_list[self.temp_ax]
        if temp_line is not None:
            # Hier vergelijken we hm mee..
            xyt = np.asarray(temp_line._xy)
            xt, yt = xyt[:, 0], xyt[:, 1]
            d = np.sqrt((xt - event.xdata) ** 2 + (yt - event.ydata) ** 2)
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
        self.line_list = [None] * self.ncols_plot
        self.point_list = [None] * self.ncols_plot
        self.axes_imshow = [None] * self.ncols_plot
        self.img_array_list = [None] * self.ncols_plot
        self.plot_array_list = [None] * self.ncols_plot


if __name__ == '__main__':
    qapp = PyQt5.QtWidgets.QApplication(sys.argv)
    ddata = '/home/bugger/Documents/Politie/PSI/klas'
    ddest = '/home/bugger/Documents/Politie/PSI/klas_proc'
    app = CropTool(ddata, ddest)
    app.show()
    qapp.exec_()
