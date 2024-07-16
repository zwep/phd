# encoding: utf-8

import numpy as np
import os
import sys

import PyQt5.QtGui
import PyQt5.QtCore
import PyQt5.QtWidgets

import matplotlib.pyplot as plt
import helper.array_transf as harray

import PyQt5.QtGui
import PyQt5.QtCore
import PyQt5.QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.backend_bases import MouseButton

from tooling.genericinterface import GenericInterface

import pathlib


class MaskInterface(GenericInterface):
    # We hope to create this tool such that we have a generic Mask-interaciton tool
    # Child classes of this will have different effect on WHAT we do with the data

    def __init__(self, data_dir, dest_dir, max_ds=10, **kwargs):
        super().__init__()

        self.debug = kwargs.get('debug')
        # Object definition
        # Denotes the active vertice
        self._ind = None
        # Below we have properties of the polygons we are creating
        self.poly = None
        self.poly_vis = True
        self.poly_collection = []
        self.verts = None
        self.line = None  # line obj

        # Indicitors if we are moving or pressing
        self.press = False
        self.move = False

        # Ax definition // plot data
        self.showverts = True
        self.max_ds = max_ds

        # Initial values for the arrays we are dealing with
        self.mask = None
        self.img_array = None
        self.left_array = None
        self.right_array = None
        self.view_id = 0

        # Used as default path when opening files
        self.data_dir = str(pathlib.Path(data_dir).expanduser())
        self.dest_dir = str(pathlib.Path(dest_dir).expanduser())

        self.mask_name_appendix = '_mask'

        # ============== PyQt5 Impl ==============
        h_layout = PyQt5.QtWidgets.QHBoxLayout(self._main)
        h_layout.addLayout(self.button_box_layout)

        # Calculates a function based off the polygon that is draw and the left array
        calc_button = self.get_push_button(name='Calculate', shortcut='C', connect=self.calculate)
        # Toggles visibility of polygon that has been created
        vis_button = self.get_push_button(name='Toggle visibility', shortcut='T', connect=self.set_vis)
        write_button = self.get_push_button(name='Writes results', shortcut='W', connect=self.write_right_array)

        self.button_box_layout.addWidget(calc_button)
        self.button_box_layout.addWidget(vis_button)
        self.button_box_layout.addWidget(write_button)

        # Create a single canvas for plotting
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2)
        if hasattr(self.axes, 'ravel'):
            self.axes = self.axes.ravel()

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(PyQt5.QtWidgets.QSizePolicy.Expanding,
                                  PyQt5.QtWidgets.QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.canvas.setFocusPolicy(PyQt5.QtCore.Qt.ClickFocus)
        self.canvas.setFocus()

        h_layout.addWidget(self.canvas)

        # Recall objects
        # Callback definition
        self.canvas.mpl_connect('draw_event', self.draw_callback)
        self.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

        if self.debug:
            print('MaskInterface initialization \t The canvas has been created')
        #
        # if init_plot:
        #     # Plot initial
        #     self.img_array = img_array
        #     self.left_array = img_array  # self.process_array(img_array)
        #     # Note that this ax will be the main axis... In this one we can write and stuff
        self.left_ax = self.axes[0]
        self.left_ax_imshow = None
        self.right_ax = self.axes[1]
        self.right_ax_imshow = None
        #
        #     self.left_ax_imshow = self.left_ax.imshow(self.left_array)
        #     self.left_title = 'Draw your region here'
        #     self.left_ax.set_title(self.left_title)
        #
        #     if self.mask is None:
        #         self.mask = np.zeros(img_array.shape)
        #
        #     self.right_array = self.mask * self.left_array
        #     self.right_ax_imshow = self.right_ax.imshow(self.right_array)
        #     self.right_title = 'Resulting mask is shown here'
        #     self.right_ax.set_title(self.right_title)
        #
        #     plt.close()
        #
        #     if self.debug:
        #         print('MaskInterface initialization \t The plots have been created')

    def key_press_callback(self, event):
        if event.key == "a":
            # Add current poly to the collection..
            self.poly_collection.append(self.poly)
            # Create new poly..
            self.poly = None
            self.line = None
            self.canvas.draw()

    def change_file(self, mod=0):
        self.view_id += mod
        file_list = os.listdir(self.data_dir)
        # Filter on files only.. using .isfile() is not viable
        # So we check if the file has an extension
        file_list = [x for x in file_list if os.path.splitext(x)[1]]

        # Filter based on what we have already written..
        n_cutoff = len(self.mask_name_appendix)
        target_file_list = [os.path.splitext(x)[0][:-n_cutoff] for x in os.listdir(self.dest_dir)]
        # TO debug stuff.. do not filter the file_list
        file_list = [x for x in file_list if os.path.splitext(x)[0] not in target_file_list]
        n_files = len(file_list)
        if n_files == 0:
            print('No more files left...')
            temp_array = np.zeros((10, 10))
            self.set_new_file(temp_array)
            return -1

        self.view_id = self.view_id % n_files  # Make sure that we can rotate through

        self.file_name = file_list[self.view_id]
        temp_array = self.load_file(self.data_dir, self.file_name)
        self.left_title = '{}'.format(str(self.file_name))
        self.right_title = '{}'.format(str(self.file_name))
        self.set_new_file(temp_array)  # Sets all the parameters to a proper default value

    def calculate(self):
        # Get the mask..
        self.mask = self.get_mask()
        self.right_array = self.left_array * self.mask
        self.update_plots()

    def write_right_array(self):
        self.mask = self.get_mask()
        if not os.path.isdir(self.dest_dir):
            os.mkdir(self.dest_dir)

        temp_file = os.path.splitext(self.file_name)[0]
        mask_path = os.path.join(self.dest_dir, temp_file + self.mask_name_appendix)
        print('Writing the mask')
        print('To file path', mask_path)
        np.save(mask_path, self.mask)

    def open_file(self):
        print('Starting open_file')
        print('Searching in ', self.data_dir)
        name, _ = PyQt5.QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', self.data_dir, "",
                                                              options=PyQt5.QtWidgets.QFileDialog.DontUseNativeDialog)
        if name:
            name, ext = os.path.splitext(name)
            self.data_dir = os.path.dirname(name)  # Can be used to skip through the files..
            self.file_name = os.path.basename(name) + ext
            self.view_id = os.listdir(self.data_dir).index(self.file_name)

            temp_array = self.load_file(self.data_dir, self.file_name)
            self.left_title = 'File {}'.format(str(self.file_name))
            self.right_title = '{}'.format(str(self.file_name))
            self.set_new_file(temp_array)  # Sets all the parameters to a proper default value

    @staticmethod
    def load_file(data_dir, file_name):
        try:
            file_dir = os.path.join(data_dir, file_name)
            print('Loading data from ', file_dir)
            file_name, ext = os.path.splitext(file_name)

            if ext.endswith('list'):
                temp_DL = mri_load.DataListImage(sel_dir=data_dir, sel_file=file_name, status=True)
                temp_image_data = temp_DL.transform_scan_data(complex=False)
                temp_array = np.squeeze(temp_image_data)

            elif ext.endswith('cpx'):
                B1Distr, data_labels = load_cpx.read_cpx_img(file_dir)
                ind_loc = data_labels.index('loc')
                n_loc = B1Distr.shape[ind_loc]
                temp_array = np.squeeze(np.take(B1Distr, n_loc//2, axis=ind_loc))

            elif ext.endswith('npy'):
                temp_array = np.load(file_dir)
            else:
                temp_array = []

            print('Loaded the file')
        except FileNotFoundError:
            print('We have not found the file...')
            temp_array = None

        return temp_array

    def set_new_file(self, img_array):
        raise NotImplementedError

    @staticmethod
    def _auto_scale(ax, new_ratio):
        # Used to get the right aspect ratio of the figure..
        # Autoscale somehow did not work. This does.

        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        cur_ratio = abs((xright - xleft) / (ybottom - ytop))
        # the abs method is used to make sure that all numbers are positive
        # because x and y axis of an axes maybe inversed.
        ax.set_aspect(cur_ratio * new_ratio)
        return ax

    def update_plots(self):
        # Check if ax_imshow exists, set_data is a lot faster
        if self.right_ax_imshow is None:
            self.right_ax_imshow = self.right_ax.imshow(self.right_array)
        else:
            self.right_ax_imshow.set_data(self.right_array)

        self.right_ax_imshow.set_clim([np.min(self.right_array), np.max(self.right_array)])
        self.right_ax.set_title(self.right_title)

        if self.left_ax_imshow is None:
            self.left_ax_imshow = self.left_ax.imshow(self.left_array)
        else:
            self.left_ax_imshow.set_data(self.left_array)

        self.left_ax_imshow.set_clim([np.min(self.left_array), np.max(self.left_array)])
        self.left_ax.set_title(self.left_title)

        self.canvas.draw()

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.left_ax.bbox)

        # Make sure that we have changed something to poly..
        if self.poly is not None:
            self.left_ax.draw_artist(self.poly)
            self.left_ax.draw_artist(self.line)

    def get_ind_under_cursor(self, event):
        'get the index of the vertex under cursor if within max_ds tolerance'
        # display coords
        if self.poly is not None:
            xy = np.asarray(self.poly.xy)
            xyt = self.poly.get_transform().transform(xy)
            xt, yt = xyt[:, 0], xyt[:, 1]
            d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
            indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
            ind = indseq[0]
            if d[ind] >= self.max_ds:
                ind = None
        else:
            ind = None
        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        ignore = not self.showverts or event.inaxes is None
        if ignore:
            print('Button press callback: Ignored')
            if self.debug:
                print('verts ', self.showverts)
                print('inaxes ', event.inaxes)
                print('button ', event.button)

            return

        self._ind = self.get_ind_under_cursor(event)
        self.press = True

        if event.button == MouseButton.RIGHT:
            if self.debug:
                print('Right click')
            self.remove_poly()
        self.update_plots()

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        ignore = not self.showverts or event.button != 1
        if ignore:
            return

        # Check that we are in the main ax..
        if event.inaxes == self.left_ax:
            # Find the index point we are closest to
            self._ind = self.get_ind_under_cursor(event)
            if event.button == MouseButton.LEFT:
                # With the left mouse button
                    if self.press and not self.move:
                        self._update_poly_add(event)
                        self._update_line()

        if self.debug:
            print('\t Released')
        self._ind = None
        self.press = False
        self.move = False
        self.canvas.draw()

    def motion_notify_callback(self, event):
        "on mouse movement"
        ignore = (not self.showverts or event.inaxes is None or
                  event.button != 1 or self._ind is None)
        if ignore:
            return
        x, y = event.xdata, event.ydata

        if self.press:
            self.move = True

        if self.debug:
            print('\t Indicator/Last vertex ', self._ind, self.last_vert_ind)

        if self._ind == 0 or self._ind == self.last_vert_ind:
            self.poly.xy[0] = x, y
            self.poly.xy[self.last_vert_ind] = x, y
        else:
            self.poly.xy[self._ind] = x, y
        self._update_line()

        self.canvas.restore_region(self.background)
        self.left_ax.draw_artist(self.poly)
        self.left_ax.draw_artist(self.line)
        self.canvas.blit(self.left_ax.bbox)
        self.canvas.draw()

    def get_mask(self):
        """
        Return image mask given by mask creator
        Here we get the mask, the values of the masked array and intensitiy sacling
        """

        h, w = self.left_array.shape
        y, x = np.mgrid[:h, :w]

        points = np.transpose((x.ravel(), y.ravel()))

        mask = np.zeros((h, w), dtype=bool)
        # Get the mask of all the combined poly-objects.
        total_poly_list = self.poly_collection + [self.poly]
        total_poly_list = [x for x in total_poly_list if x]

        if self.debug:
            print('\t\t Get mask ')
            for i, i_poly in enumerate(total_poly_list):
                print('stored verts ', i)
                print('          \t', i_poly.xy)

        if self.verts is not None:
            if len(self.verts) > 1:
                for i, i_poly in enumerate(total_poly_list):
                    path = Path(i_poly.xy)
                    temp_mask = path.contains_points(points)
                    temp_mask = temp_mask.reshape(h, w).astype(bool)
                    if self.debug:
                        print('stored points ', i)
                        print('            \t', np.sum(temp_mask))
                    mask += temp_mask
            else:
                mask = None
        else:
            mask = None

        return mask

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        # Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def remove_poly(self):

        if self.poly is None:
            # If we did not have something.. but have something in store, we trieve that.
            if self.poly_collection:
                self.poly = self.poly_collection[-1]
                self.poly_collection.pop()
                temp_x, temp_y = zip(*self.poly.xy)
                self.line = plt.Line2D([temp_x], [temp_y], color='black', marker='o', mfc='r', alpha=0.8, animated=True)
                self.left_ax.add_line(self.line)
                self.left_ax.add_patch(self.poly)
        else:
            # If we had something.. we remove it..
            self.poly = None
            self.line = None

        self.mask = self.get_mask()

    def _update_poly_add(self, event):
        if self.poly is None:
            if self.debug:
                print('First event', event.xdata, event.ydata)
            self.line = plt.Line2D([event.xdata], [event.ydata], color='black', marker='o', mfc='r', alpha=0.8,
                                   animated=True)
            self.poly = Polygon([[event.xdata, event.ydata]], animated=True, fc='y', ec='none', alpha=0.4)
            self.left_ax.add_patch(self.poly)
            self.left_ax.add_line(self.line)
            self.left_ax.set_clip_on(False)
            self.poly.add_callback(self.poly_changed)

        elif len(self.poly.xy) == 1:
            if self.debug:
                print('Adding second point...', event.xdata, event.ydata)
            temp = np.vstack([self.poly.xy, [event.xdata, event.ydata]])
            self.poly.xy = temp
        else:
            if self.debug:
                print('Adding one point...', event.xdata, event.ydata)
            xys = self.poly.get_transform().transform(self.poly.xy)

            p = event.x, event.y  # cursor coords
            temp_dist = []
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                # Distance to segment points..
                temp = harray.point_to_line_dist(np.array(p), np.vstack([s0, s1]))
                temp_dist.append(temp)

            i_min = np.argmin(temp_dist)
            self.poly.xy = np.insert(self.poly.xy, i_min+1, [event.xdata, event.ydata], axis=0)

        # We want to visualize where the masks are..
        # We used to do this with add_patch or add_collections.. but that is kinda buggy somehow.
        # This is a quick-fix: We just overlay the mask on the plot..
        temp_mask = self.get_mask()
        if temp_mask is not None:
            self.mask = temp_mask
            self.update_plots()

    def _update_line(self):
        # save verts because polygon gets deleted when figure is closed
        self.verts = self.poly.xy
        self.last_vert_ind = len(self.poly.xy) - 1
        self.line.set_data(zip(*self.poly.xy))

    def set_vis(self):
        if self.poly is not None:
            self.poly.set_visible(not self.poly_vis)
            self.line.set_visible(not self.poly_vis)
            self.canvas.draw()
            self.poly_vis = not self.poly_vis

    @staticmethod
    def _isfloat(value):
        try:
            return float(value)

        except ValueError:
            return -1


if __name__ == "__main__":
    A = (100 * np.random.rand(100, 100)).astype(int)
    qapp = PyQt5.QtWidgets.QApplication(sys.argv)
    app = MaskInterface(A, debug=True, add_layout_widget=True)
    app.show()
    qapp.exec_()
