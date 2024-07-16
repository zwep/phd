import math
import sys
import os

import skimage.transform as sktransf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.backend_bases import MouseButton
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import matplotlib.gridspec as gridspec
# Deze moet je blijven importeren
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.collections as mcol
import itertools
import helper.misc as hmisc
import helper.array_transf as harray
import scipy.optimize


class MaskCreatorTemplate:
    """An interactive polygon editor.
    Parameters
    ----------
    img_array: numpy array
    """

    def __init__(self, img_array, max_ds=10, initial_mask=None, **kwargs):
        # Initial mask should be a polygon object
        self.debug = kwargs.get('debug')
        self.different_area = kwargs.get('different_area', False)
        main_title = kwargs.get('main_title', 'Double click and drag a point to move it')
        left_title = kwargs.get('left_title', 'Draw your region here')
        right_title = kwargs.get('right_title', 'Resulting mask is shown here')
        # Object definition
        self._ind = None  # the active vert
        self.poly = None  # Poly obj
        self.line = None  # line obj
        self.poly_collection = []

        self.cur_poly_ind = 0
        self.img_array = img_array
        self.plot_array = self.process_array(img_array)

        self.press = False
        self.move = False

        # Ax definition // plot data
        self.showverts = True
        self.max_ds = max_ds
        fig, ax = plt.subplots(1, 2)
        canvas = fig.canvas
        self.z00 = None
        fig.suptitle(main_title)
        self.main_ax = ax[0]
        self.main_ax.imshow(self.plot_array)
        self.main_ax.set_title(left_title)

        self.mask_ax = ax[1]
        self.mask = np.zeros(self.plot_array.shape)  # Dummy start..
        self.mask_ax_imshow = self.mask_ax.imshow(self.mask)
        self.mask_ax.set_title(right_title)

        # Callback definition
        self.canvas = canvas
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

        if initial_mask is not None:
            coords = initial_mask.xy
            self.poly = initial_mask
            self.line = plt.Line2D([coords[:, 0]], [coords[:, 1]], color='black', marker='o', mfc='r', alpha=0.8, animated=True)
            self.poly_collection = []
            self.main_ax.add_patch(self.poly)
            self.main_ax.add_line(self.line)
            plt.draw()

    @staticmethod
    def process_array(img_array):
        raise NotImplementedError

    @staticmethod
    def calculate(mask, image_array):
        raise NotImplementedError

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.main_ax.bbox)

        if self.poly is not None:
            self.main_ax.draw_artist(self.poly)
            self.main_ax.draw_artist(self.line)

            self.mask, self.masked_array, mask_varray = self.get_mask(self.img_array)
            self.mask_ax_imshow.set_data(self.masked_array)
            self.mask_ax_imshow.set_clim(mask_varray)

            if self.debug:
                print('Drawing polygon/line ')
                print('Content of poly collection ', self.poly_collection)

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

    def key_press_callback(self, event):

        if event.key == "h":
            print(event.key, 'woo did it. We pressed qh')
            # HEel vreemd.
            # Toevoegen. Prima, visluzeren.. nee.
            # This should mean.. new Polygon..
            self.poly_collection.append(self.poly)
            print('Stored poly ', self.poly.xy)
            p_col = mcol.PatchCollection(self.poly_collection, alpha=0.5)

            print(self.main_ax.dataLim.bounds)
            # if self.z00:
            #     self.z00.remove()
            # self.z00 = self.main_ax.add_collection(p_col)
            # self.main_ax.add_patch(self.poly_collection[0])
            print(self.main_ax.dataLim.bounds)
            for i_poly in self.poly_collection:
                print(i_poly.get_visible())
                self.patch00 = self.main_ax.add_patch(i_poly)
                self.main_ax.autoscale_view()

            self.cur_poly_ind += 1
            self.poly = None
            self.line = None
            self.canvas.draw()
        if event.key == 'd':
            self.canvas.draw()

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        ignore = not self.showverts or event.inaxes is None
        if ignore:
            print('Ignored...')
            print('verts ', self.showverts)
            print('inaxes ', event.inaxes)
            print('button ', event.button)
            return
        #
        self._ind = self.get_ind_under_cursor(event)
        self.press = True

        if self.debug:
            print(event.button)
        if event.button == MouseButton.RIGHT:
            print('Removed current polygon')
            self.remove_poly()

        self.canvas.draw()

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        ignore = not self.showverts or event.button != 1
        if ignore:
            return

        # Check that we are in the main ax..
        if event.inaxes == self.main_ax:
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
        self.main_ax.draw_artist(self.poly)
        self.main_ax.draw_artist(self.line)
        self.canvas.blit(self.main_ax.bbox)
        self.canvas.draw()

    def get_mask(self, image_array):
        """
        Return image mask given by mask creator
        Here we get the mask, the values of the masked array and intensitiy sacling
        """

        h, w = self.plot_array.shape
        y, x = np.mgrid[:h, :w]
        points = np.transpose((x.ravel(), y.ravel()))

        # total_mask = np.zeros((h, w), dtype=bool)
        total_mask = np.zeros((h, w), dtype=int)
        total_poly_list = self.poly_collection + [self.poly]
        if self.debug:
            print('\t\t Get mask ')
            print('current verts \t', self.poly.xy)
            for i, i_poly in enumerate(total_poly_list):
                print('stored verts ', i)
                print('          \t', i_poly.xy)

        for i, i_poly in enumerate(total_poly_list):
            path = Path(i_poly.xy)
            mask = path.contains_points(points)
            # mask = mask.reshape(h, w).astype(bool)
            mask = mask.reshape(h, w)
            if self.different_area:
                total_mask += mask * (1 + i) ** 2
            else:
                total_mask += mask

        masked_array, varray = self.calculate(total_mask, image_array)
        return total_mask, masked_array, varray

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        # Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def remove_poly(self):
        # Here we should pay attention to remove ALL polygons
        self.poly = None
        self.line = None
        self.canvas.draw()
        # Now recover from the last
        self.cur_poly_ind -= 1
        self.poly_collection.pop()
        if self.poly_collection:
            self.poly = self.poly_collection[-1]
            temp_x, temp_y = zip(*self.poly.xy)
            self.line = plt.Line2D([temp_x], [temp_y], color='black', marker='o', mfc='r', alpha=0.8, animated=True)
            self.main_ax.add_line(self.line)
        else:
            self.poly = None
            self.line = None

        self.canvas.draw()

    def _update_poly_add(self, event):
        # Here we add a poly..
        # So the "New" poly could just add the current poly to a list..
        # And start making a new poly..
        # But we should visualize ALL polygons in that list thing.
        # SO we should have one object to which we add the new events..
        # ANd this object should be updated to the current polygon_stored_List..
        # VIsualization goes over this stored list thing...
        if self.poly is None:
            print('First event', event.xdata, event.ydata)
            self.line = plt.Line2D([event.xdata], [event.ydata], color='black', marker='o', mfc='r', alpha=0.8,
                                   animated=True)
            self.poly = Polygon([[event.xdata, event.ydata]], animated=True, fc='y', ec='none', alpha=0.4)
            self.main_ax.add_patch(self.poly)
            self.main_ax.add_line(self.line)
            # self.main_ax.set_clip_on(False)
            # self.poly.add_callback(self.poly_changed)  # Why is this...

        elif len(self.poly.xy) == 1:
            print('Adding second point...', event.xdata, event.ydata)
            temp = np.vstack([self.poly.xy, [event.xdata, event.ydata]])
            self.poly.xy = temp
            ind_poly = self.cur_poly_ind
            # self.poly_collection[ind_poly].xy = temp
        else:
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
            ind_poly = self.cur_poly_ind
            # self.poly_collection[ind_poly].xy = np.insert(self.poly.xy, i_min+1, [event.xdata, event.ydata], axis=0)

    def _update_line(self):
        # save verts because polygon gets deleted when figure is closed
        self.verts = self.poly.xy
        self.last_vert_ind = len(self.poly.xy) - 1
        self.line.set_data(zip(*self.poly.xy))


class MaskCreator(MaskCreatorTemplate):

    def calculate(self, mask, image_array):
        masked_array = mask * self.process_array(img_array=image_array)
        masked_varray = [np.min(masked_array), np.max(masked_array)]
        return masked_array, masked_varray

    @staticmethod
    def process_array(img_array):
        if img_array.ndim == 4:
            res = np.abs(img_array.sum(axis=0).sum(axis=0))
        elif img_array.ndim == 3:
            res = np.abs(np.sum(img_array, axis=0))
        elif img_array.ndim == 2:
            res = np.abs(img_array)
        return res

    def __mul__(self, other):
        return other * self.mask


class ListPlot:
    def __init__(self, image_list, **kwargs):
        figsize = kwargs.get('figsize', (10, 10))
        fignum = kwargs.get('fignum')
        dpi = kwargs.get('dpi', 300)

        self.title_string = kwargs.get('title', "")
        self.sub_title = kwargs.get('subtitle', None)
        self.cbar_ind = kwargs.get('cbar', False)
        self.cbar_round_n = kwargs.get('cbar_round_n', 2)
        self.cmap = kwargs.get('cmap', 'gray')

        self.vmin = kwargs.get('vmin', None)
        self.ax_off = kwargs.get('ax_off', False)
        self.augm_ind = kwargs.get('augm', None)
        self.aspect_mode = kwargs.get('aspect', 'equal')
        self.start_square_level = kwargs.get('start_square_level', 8)
        self.col_row = kwargs.get('col_row', None)
        self.sub_col_row = kwargs.get('sub_col_row', None)

        self.proper_scaling = kwargs.get('proper_scaling', False)
        self.proper_scaling_patch_shape = kwargs.get('proper_scaling_patch_shape', 128)
        self.proper_scaling_stride = kwargs.get('proper_scaling_stride', 64)
        if self.proper_scaling is True:
            if self.vmin is None:
                if isinstance(image_list, list):
                    self.vmin = []
                    temp_img_list = []
                    for i_array in image_list:
                        temp_vmin = [(0, harray.get_proper_scaled_v2(np.abs(x) if np.iscomplexobj(x) else x, patch_shape=self.proper_scaling_patch_shape,
                                                                     stride=self.proper_scaling_stride)) for x in i_array]
                        temp_img = harray.scale_minmax(np.abs(i_array) if np.iscomplexobj(i_array) else i_array, axis=(-2, -1))
                        self.vmin.append(temp_vmin)
                        temp_img_list.append(temp_img)
                    image_list = temp_img_list
                else:
                    self.vmin = [(0, harray.get_proper_scaled_v2(x, patch_shape=self.proper_scaling_patch_shape, stride=self.proper_scaling_stride)) for x in image_list]
                    image_list = harray.scale_minmax(image_list, axis=(-2, -1))
            else:
                print('Vmin is set AND proper scaling is turned on. Defaulting to vmin')

        self.wspace = kwargs.get('wspace', 0.1)
        self.hspace = kwargs.get('hspace', 0.1)

        self.debug = kwargs.get('debug', False)

        self.figure = plt.figure(fignum, figsize=figsize, dpi=dpi)
        self.figure.suptitle(self.title_string)
        self.canvas = self.figure.canvas
        # Used to go from positive to negative scaling
        self.epsilon = 0.001

        # Only when we have an numpy array
        if isinstance(image_list, np.ndarray):
            # With just two dimensions..
            if image_list.ndim == 2:
                # Add one..
                image_list = image_list[np.newaxis]

        if self.col_row is not None:
            # If this is selected.. then we expect a list of 3D arrays...
            # Is that desireable?
            n_col, n_row = self.col_row
        else:
            n_col, n_row = (1, len(image_list))
        self.gs0 = gridspec.GridSpec(n_row, n_col, figure=self.figure)
        self.gs0.update(wspace=self.wspace, hspace=self.hspace)  # set the spacing between axes.
        self.gs0.update(top=1. - 0.5 / (n_row + 1), bottom=0.5 / (n_row + 1))
        # left = 0.5 / (ncol + 1), right = 1 - 0.5 / (ncol + 1))

        if self.debug:
            print("Status of loaded array")
            print("\tNumber of rows//length of image list ", n_row)
            if hasattr(image_list, 'ndim'):
                print("\tDimension of image list", image_list.ndim)
            if hasattr(image_list[0], 'ndim'):
                print("\tDimension of first image list element ", image_list[0].ndim)

        self.ax_list, self.ax_imshow_list, self.ax_cbar_list = self.plot_3d_list(image_list)
        self.ax_list = list(itertools.chain(*self.ax_list))
        self.ax_imshow_list = list(itertools.chain(*self.ax_imshow_list))
        self.ax_cbar_list = list(itertools.chain(*self.ax_cbar_list))

        self.press_indicator = False
        self.press_position = None
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('motion_notify_event', self.move_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)

    def savefig(self, name, home=True, pad_inches=0.0, bbox_inches='tight'):
        # Default values for pad_inches is 0.1
        # Default values for bbox_inches is None

        if name.endswith(".png"):
            name = name[:-4]

        if home:
            dest_path = os.path.expanduser(f"~/{name}.png")
        else:
            dest_path = name

        print(f'Saving image to {dest_path}')
        self.figure.savefig(dest_path, bbox_inches=bbox_inches, pad_inches=pad_inches)

    def button_press_callback(self, input):
        if input.inaxes in self.ax_list:
            index_current_ax = self.ax_list.index(input.inaxes)
            self.temp_ax = self.ax_imshow_list[index_current_ax]

            # Reset colorbar
            if input.button == MouseButton.RIGHT:
                temp_array = self.temp_ax.get_array()
                temp_cbar = self.ax_cbar_list[index_current_ax]
                reset_clim = [temp_array.min(), temp_array.max()]
                self.temp_ax.set_clim(reset_clim)
                if temp_cbar:
                    temp_cbar.set_ticks(reset_clim)
                self.canvas.draw()
            elif input.button == MouseButton.LEFT:
                self.press_indicator = True
                self.press_position = {'x': input.xdata, 'y': input.ydata}
                self.press_clim = list(self.temp_ax.get_clim())
            else:
                print('Unknown button pressed ', input.button, input)

            if self.debug:
                print("You have clicked", input, self.ax_list.index(input.inaxes))
                print("Previous clim ", self.press_clim)

    def move_callback(self, input):
        if input.inaxes in self.ax_list:
            if self.press_indicator:
                current_position = {'x': input.xdata, 'y': input.ydata}
                index_current_ax = self.ax_list.index(input.inaxes)

                size_x, size_y = self.temp_ax.get_size()
                distance_x = (current_position['x'] - self.press_position['x']) / size_x
                distance_y = (current_position['y'] - self.press_position['y']) / size_y

                if self.press_clim[1] < 0:
                    distance_x = 1 - distance_x
                else:
                    distance_x = 1 + distance_x

                if self.press_clim[0] < 0:
                    distance_y = 1 - distance_y
                else:
                    distance_y = 1 + distance_y

                if np.abs(self.press_clim[1]) < self.epsilon:
                    if self.press_clim[1] > 0:
                        self.press_clim[1] = -2 * self.epsilon
                    else:
                        self.press_clim[1] = 2 * self.epsilon

                if np.abs(self.press_clim[0]) < self.epsilon:
                    if self.press_clim[0] > 0:
                        self.press_clim[0] = - 2 * self.epsilon
                    else:
                        self.press_clim[0] = 2 * self.epsilon

                max_clim = self.press_clim[1] * distance_x
                min_clim = self.press_clim[0] * distance_y

                new_clim = [min_clim, max_clim]
                self.temp_ax.set_clim(new_clim)

                if self.cbar_ind:
                    temp_cbar = self.ax_cbar_list[index_current_ax]
                    temp_cbar.set_ticks(new_clim)
                self.canvas.draw()

    def button_release_callback(self, input):
        # if input.inaxes:
        self.press_indicator = False
        self.press_position = None

    def plot_3d_list(self, image_list):
        # Input of either a 2d list of np.arrays.. or a 3d list of np.arrays..
        ax_list = []
        ax_imshow_list = []
        ax_cbar_list = []
        for i, i_gs in enumerate(self.gs0):
            if i >= len(image_list):
                print(f"STOP - length is violated {i} >= {len(image_list)}")
                break
            sub_ax_list = []
            sub_ax_imshow_list = []
            sub_cbar_list = []
            temp_img = image_list[i]
            # Here we split between having a numpy array
            # or a list...
            if hasattr(temp_img, 'ndim') and hasattr(temp_img, 'shape') and hasattr(temp_img, 'reshape'):
                if temp_img.ndim == 4:
                    n_sub_col = temp_img.shape[0]
                    n_sub_row = temp_img.shape[1]
                    # With this we want to prevent plotting a 3D array in the next step
                    # But avoid this with rgb images..
                    if self.cmap == 'rgb':
                        # Make atleast check something I guess
                        assert temp_img.shape[-1] == 3, "Last image dimension is not equal to three"
                    else:
                        #
                        temp_img = temp_img.reshape((n_sub_col * n_sub_row,) + temp_img.shape[2:])
                elif temp_img.ndim == 3:
                    n_sub_col = temp_img.shape[0]
                    if n_sub_col > self.start_square_level:
                        n_sub_col, n_sub_row = hmisc.get_square(n_sub_col)
                        print('Using sub col, sub row:', n_sub_col, n_sub_row)
                    else:
                        n_sub_row = 1
                elif temp_img.ndim == 2:
                    temp_img = temp_img[np.newaxis]
                    n_sub_col = 1
                    n_sub_row = 1
                else:
                    print('Unknown image dimension: ', temp_img.shape)
                    return
            else:
                n_sub_col = len(temp_img)
                n_sub_row = 1

            # If we have configured sub col row.. we want to impose that now
            if self.sub_col_row is not None:
                # If this is selected.. then we expect a list of 3D arrays...
                # Is that desireable?
                n_sub_col, n_sub_row = self.sub_col_row

            if self.debug:
                print(f"\tVariable temp_image is has length {len(temp_img)} and shape {temp_img.shape}")
                print("\tThe number of col/row is set to", n_sub_col, n_sub_row)

            # If we want to specifcy the vmin per list item.. we can do that here..
            if isinstance(self.vmin, list):
                sel_vmin = self.vmin[i]
            else:
                sel_vmin = self.vmin

            for j, ii_gs in enumerate(i_gs.subgridspec(n_sub_row, n_sub_col, wspace=self.wspace, hspace=self.hspace)):
                # Do not continue plotting when we are exceeding the number of things to plot
                # This avoids trying to plot stuff in an axes when everything is already plotted.
                if j >= len(temp_img):
                    print(f"STOP - length is violated {j} >= {len(temp_img)}")
                    break
                # Hacky way to fix the list in list vmin specification
                if isinstance(sel_vmin, list):
                    sel_sel_vmin = sel_vmin[j]
                else:
                    sel_sel_vmin = sel_vmin

                ax = self.figure.add_subplot(ii_gs)
                if self.augm_ind:
                    plot_img = eval('{fun}({var})'.format(fun=self.augm_ind, var=str('temp_img[j]')))
                    if 'angle' in self.augm_ind:
                        sel_sel_vmin = (-np.pi, np.pi)
                else:
                    if np.iscomplexobj(temp_img[j]):
                        # If we did not choose anything, default to 'abs'
                        plot_img = np.abs(temp_img[j])
                    else:
                        plot_img = temp_img[j]

                if self.debug:
                    print(f'Image id: {i} - shape of temp image {temp_img.shape}', end=' \t|\n')
                    print(f'Plot id: {j} - shape of plot image {plot_img.shape}', end=' \t|\n')

                if self.cmap == 'rgb':
                    # For this to work.. there are some prequisites..
                    # First: shape of array should be: (1, nx, ny, 3)
                    # Second: Array should be given like [plot_array]
                    # Third: Give a sub_col_row with a value
                    # Last: of course, set cmap=rgb
                    map_ax = ax.imshow(plot_img, vmin=sel_sel_vmin, aspect=self.aspect_mode)
                elif isinstance(self.cmap, list):
                    map_ax = ax.imshow(plot_img, vmin=sel_sel_vmin, aspect=self.aspect_mode, cmap=self.cmap[i][j])
                else:
                    map_ax = ax.imshow(plot_img, vmin=sel_sel_vmin, aspect=self.aspect_mode, cmap=self.cmap)

                if self.cbar_ind:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    temp_cbar = plt.colorbar(map_ax, cax=cax)
                    if sel_sel_vmin is None:
                        vmin_temp = [plot_img.min(), plot_img.max()]
                        vmin_temp = list(map(float, vmin_temp))
                        map_ax.set_clim(vmin_temp)
                        temp_cbar.set_ticks([np.round(x, self.cbar_round_n) for x in vmin_temp])
                    else:
                        map_ax.set_clim(sel_sel_vmin)
                        # temp_cbar.set_ticks([np.round(x, self.cbar_round_n) for x in sel_sel_vmin])
                else:
                    temp_cbar = None

                if self.sub_title is not None:
                    ax.set_title(self.sub_title[i][j])
                if self.ax_off:
                    ax.set_axis_off()

                sub_ax_list.append(ax)
                sub_ax_imshow_list.append(map_ax)
                sub_cbar_list.append(temp_cbar)

            ax_list.append(sub_ax_list)
            ax_imshow_list.append(sub_ax_imshow_list)
            ax_cbar_list.append(sub_cbar_list)

        return ax_list, ax_imshow_list, ax_cbar_list


class SlidingPlot:
    def __init__(self, x, slider_name=None, fignum=None, title='', ax_3d=False, **kwargs):
        """
        Is able to easily plot n-dimensional data, where the last two are used to plt.imshow()
        The others become Slider axes that can be issued.

        :param x: Can be an any dimensional numpy array
        :param slider_name:
        :param fignum:
        :param title:
        """

        # Define initial variables
        self.init_slice = 1
        self.init_data = x
        self.x_shape = x.shape
        self.ax_3d = ax_3d

        # Variables used for plotting
        self.plot_data = np.abs(np.copy(x))
        vmin = kwargs.get('vmin')
        if isinstance(vmin, tuple):
            self.varray = vmin
        else:
            self.varray = [np.min(self.plot_data), np.max(self.plot_data)]

        # Figure/Axes definitions
        # !! Hier zou dus een andere ax moeten zinj als t 3d wordt...
        if self.ax_3d:
            self.fig = plt.figure(num=fignum)
            self.ax = self.fig.gca(projection='3d')
        else:
            self.fig, self.ax = plt.subplots(num=fignum)
            plt.subplots_adjust(left=0.25, top=0.8, bottom=0.2)
            divider = make_axes_locatable(self.ax)
            self.axcolorbar = divider.append_axes('right', size='5%', pad=0.05)

        self.axcolor = 'lightgoldenrodyellow'
        self.delta_ax = 0.05
        self.n_slider = len(self.x_shape) - 2
        self.max_slider = [x-1 for x in self.x_shape[:-2]]

        if slider_name is None:
            self.slider_name = ['Axes {}'.format(str(i)) for i in range(self.n_slider)]
        else:
            self.slider_name = slider_name

        # Create slider
        self.slider_value = tuple(np.zeros(self.n_slider, dtype=int))
        self.sliderax_list = [self.fig.add_axes([0.05, 0.25 + self.delta_ax * i, 0.1, 0.03], facecolor=self.axcolor) for
                              i in range(self.n_slider)]
        # Needed to have manuall buttons for the slider axes...
        self.sliderax_left_button_list = [
            self.fig.add_axes([0.2, 0.25 + self.delta_ax * i, 0.05, 0.03], facecolor=self.axcolor) for i in
            range(self.n_slider)]
        self.sliderax_right_button_list = [
            self.fig.add_axes([0.25, 0.25 + self.delta_ax * i, 0.05, 0.03], facecolor=self.axcolor) for i in
            range(self.n_slider)]

        self.slider_list = [Slider(self.sliderax_list[i], self.slider_name[i],
                                   self.slider_value[i],
                                   self.max_slider[i],
                                   valinit=0) for i in range(self.n_slider)]

        self.slider_left_button_list = [Button(self.sliderax_left_button_list[i], '<',
                                               color=self.axcolor, hovercolor='0.975')
                                        for i in range(self.n_slider)]
        self.slider_right_button_list = [Button(self.sliderax_right_button_list[i], '>',
                                               color=self.axcolor, hovercolor='0.975')
                                         for i in range(self.n_slider)]
        # Act upon change for all sliders
        [x.on_changed(self.update) for x in self.slider_list]
        [x.on_clicked(lambda x: self.update_button(x, -1)) for x in self.slider_left_button_list]
        [x.on_clicked(lambda x: self.update_button(x, 1)) for x in self.slider_right_button_list]

        # Create rescale button
        self.rescaleax = plt.axes([0.05, 0.2, 0.1, 0.03])
        self.button_rescale = Button(self.rescaleax, 'Resscale', color=self.axcolor, hovercolor='0.975')
        self.button_rescale.on_clicked(self.rescale)

        # Create reset button
        self.resetax = plt.axes([0.05, 0.17, 0.1, 0.03])
        self.button = Button(self.resetax, 'Reset', color=self.axcolor, hovercolor='0.975')
        self.button.on_clicked(self.reset)

        # Create data augmnent func
        self.rax = plt.axes([0.05, 0.02, 0.1, 0.15], facecolor=self.axcolor)
        self.radio = RadioButtons(self.rax, ('abs', 'angle', 'imag', 'real', 'log-abs'), active=0)
        self.radio.on_clicked(self.datatype)

        # Plot the data
        # self.slider.drawon = False
        if self.ax_3d:
            n_y, n_x = self.plot_data.shape[-2:]
            self.X, self.Y = np.meshgrid(range(n_x), range(n_y))
            self.aximshow = self.ax.plot_surface(self.X, self.Y, self.plot_data[self.slider_value], cmap=cm.coolwarm)

            self.scalarmap = cm.ScalarMappable(norm=mcolors.Normalize(*self.varray), cmap=cm.coolwarm)
            self.colorbarobj = self.fig.colorbar(self.scalarmap, ax=self.ax)
        else:
            self.aximshow = self.ax.imshow(self.plot_data[self.slider_value], vmin=self.varray, cmap='gray')

            self.scalarmap = cm.ScalarMappable(norm=mcolors.Normalize(*self.varray), cmap='gray')
            self.colorbarobj = self.fig.colorbar(self.scalarmap, ax=self.aximshow, cax=self.axcolorbar)

        self.fig.suptitle(title)

    def _aug_data(self, label):
        if label == 'real':
            temp_data = np.real(self.init_data)
        elif label == 'imag':
            temp_data = np.imag(self.init_data)
        elif label == 'abs':
            temp_data = np.abs(self.init_data)
        elif label == 'angle':
            temp_data = np.angle(self.init_data)
        elif label == 'log-abs':
            temp_data = np.arcsinh(np.abs(self.init_data))
        else:
            temp_data = -1

        return temp_data

    def update_button(self, value, change_value):
        # Left button is pressed
        # print('Pressed button', value, change_value)
        if change_value == -1:
            axes_ind = [i for i, x in enumerate(self.sliderax_left_button_list) if x == value.inaxes][0]
            # print('Found axes ind ', axes_ind)
        # Right button is pressed
        elif change_value == 1:
            axes_ind = [i for i, x in enumerate(self.sliderax_right_button_list) if x == value.inaxes][0]
            # print('Found axes ind ', axes_ind)
        else:
            axes_ind = 0

        new_slider_value = self.slider_value[axes_ind] + change_value
        new_slider_value = max(new_slider_value, 0)
        new_slider_value = min(new_slider_value, self.max_slider[axes_ind])

        self.slider_list[axes_ind].set_val(new_slider_value)
        self.update(None)

    def update(self, value):
        # unpack list and use that to subset data
        self.slider_value = tuple([int(x.val) for x in self.slider_list])
        [x.valtext.set_text('{}'.format(int(x.val))) for x in self.slider_list]
        if self.ax_3d:
            self.ax.cla()
            self.ax.plot_surface(self.X, self.Y, self.plot_data[self.slider_value], cmap=cm.coolwarm)
        else:
            # Update slider setting
            self.aximshow.set_data(self.plot_data[self.slider_value])

        # Draw everything
        self.colorbarobj.draw_all()
        self.fig.canvas.draw()

    def reset(self, event):
        [x.reset() for x in self.slider_list]
        self.radio.set_active(0)

    def datatype(self, label):
        self.plot_data = self._aug_data(label)
        self.update(None)

    def rescale(self, event):
        # temp_data = self.plot_data[self.slider_value]
        # self.plot_data[self.slider_value] = temp_data / np.max(temp_data)
        # Update clim
        min_plot = np.min(self.plot_data[self.slider_value])
        max_plot = np.max(self.plot_data[self.slider_value])
        self.varray = [min_plot, max_plot]
        self.aximshow.set_clim(self.varray)
        self.scalarmap.set_clim(self.varray)
        self.update(None)


class ComparePlot:
    # Wat hier nog te doen...
    # -> zorg ervoor dat je door locaties kan scrollen
    # -> zorg voor een save button (van zowel de b1 shim series als die ander)
    # -> Split dit ding op in meerdere units..
    # Eentje om de UI helemaal te fixen
    # Een ander om de specifieke B1Shim dingen te doen.
    """
        Visualizatiee
            Abs som van alle axes van de voor laatste twee

        Write
            Opslaan van X, Y, complex chan per coil.
            Dit betekend dus ook dat je ze op een manier moet samen voegen.
                Bekijk van de complexe plaatjes de fases over de receive channel voor één transmit kanaal
                Je zou ze eruit kunnen delen... dit lijkt mij dan te gebeuren per pixel waarde..
                    Check wat het fmin search vind.
                        Stel daar ook de zoek ruimte van vast.. 0, ..., 2pi
                    Is dit het zelfde als het gemiddelde van de fases per plaatje.
                Je kan ook checken hoe de fase verdeling is..

    """

                # zorg er ook voor dat je de COIL data opslaat. En niet het mooie plaatje)
    # ->

    def __init__(self, left_img, right_img, dest_dir=None, file_prefix="", train=True, **kwargs):
        """

        """
        self.debug = kwargs.get('debug')
        self.left_img = left_img
        self.right_img = right_img
        self.left_plot_img = self.process_array(left_img, side='left')
        self.right_plot_img = self.process_array(right_img, side='right')
        self.train = train

        # Typical..
        self.l_loc, self.l_h, self.l_w = self.left_plot_img.shape
        self.r_h, self.r_w = self.right_plot_img.shape

        self.dest_dir = dest_dir
        self.file_prefix = file_prefix
        fignum = kwargs.get('fignum')

        # Figure/Axes definitions
        self.fig = plt.figure(num=fignum)
        img_gs = gridspec.GridSpec(1, 3, figure=self.fig)
        side_view_gs = img_gs[1].subgridspec(2, 1)
        self.left_img_ax = self.fig.add_subplot(img_gs[0])
        self.right_img_ax = self.fig.add_subplot(img_gs[2])
        self.top_ax = self.fig.add_subplot(side_view_gs[0])
        self.bottom_ax = self.fig.add_subplot(side_view_gs[1])
        self.axcolor = 'lightgoldenrodyellow'
        self.delta_ax = 0.05
        self.max_shift = self.l_w - self.r_w

        # Add slider that allows to shift through the position
        self.slider_value = 0
        self.sliderax = self.fig.add_axes([0.05, 0.25 + self.delta_ax * 1, 0.1, 0.03], facecolor=self.axcolor)
        self.slider_ax = Slider(self.sliderax, 'Shift', valmin=0, valmax=self.max_shift, valinit=self.slider_value)
        self.slider_ax.on_changed(self.update)

        # Add slider that allows to shift through locations
        self.loca_value = 0
        self.locaax = self.fig.add_axes([0.05, 0.25 + self.delta_ax * 2, 0.1, 0.03], facecolor=self.axcolor)
        self.slider_loca = Slider(self.locaax, 'Slice', valmin=0, valmax=self.l_loc-1, valinit=self.loca_value)
        self.slider_loca.on_changed(self.update)

        # Add a flip button
        self.ind_flip = False
        self.flipax = self.fig.add_axes([0.05, 0.25 + self.delta_ax * 3, 0.1, 0.03], facecolor=self.axcolor)
        self.button_flip = Button(self.flipax, 'Flip')
        self.button_flip.on_clicked(self.flip)

        # Add a scale button
        self.ind_scale = False
        self.scaleax = self.fig.add_axes([0.05, 0.25 + self.delta_ax * 4, 0.1, 0.03], facecolor=self.axcolor)
        self.button_scale = Button(self.scaleax, 'Scale')
        self.button_scale.on_clicked(self.scale)

        # Add a allign button
        self.alignax = self.fig.add_axes([0.05, 0.25 + self.delta_ax * 5, 0.1, 0.03], facecolor=self.axcolor)
        self.button_align = Button(self.alignax, 'Align')
        self.button_align.on_clicked(self.align)

        # Add slider that allows to change width
        self.rw_value = self.r_w
        self.rwax = self.fig.add_axes([0.05, 0.25 + self.delta_ax * 6, 0.1, 0.03], facecolor=self.axcolor)
        self.slider_rw = Slider(self.rwax, 'Width', valmin=0, valmax=self.r_w, valinit=self.rw_value)
        self.slider_rw.on_changed(self.update)

        # Add slider that allows to shift through the position
        self.r_slider_value = 0
        self.r_sliderax = self.fig.add_axes([0.05, 0.25 + self.delta_ax * 7, 0.1, 0.03], facecolor=self.axcolor)
        self.r_slider_ax = Slider(self.r_sliderax, 'Shift', valmin=0, valmax=self.max_shift, valinit=self.r_slider_value)
        self.r_slider_ax.on_changed(self.update)

        # Add a write button
        self.buttonax = self.fig.add_axes([0.05, 0.25 + self.delta_ax * 8, 0.1, 0.03], facecolor=self.axcolor)
        self.button_write = Button(self.buttonax, 'Write')
        self.button_write.on_clicked(self.write_file)

        # Plot data
        temp = self.left_plot_img[self.loca_value, :, self.slider_value:(self.slider_value+self.r_w)]
        self.left_aximshow = self.left_img_ax.imshow(temp)
        self.left_img_ax.set_title('Left plot')

        # Plot plot
        left_profile_x = np.sum(temp, axis=0)
        left_profile_y = np.sum(temp, axis=1)
        right_profile_x = np.sum(self.right_plot_img, axis=0)
        right_profile_y = np.sum(self.right_plot_img, axis=1)
        self.left_top_ax_x, = self.top_ax.plot(left_profile_x, 'r', alpha=0.5, label='left plot')
        self.right_top_ax_x, = self.top_ax.plot(right_profile_x, 'b', alpha=0.5, label='right plot')
        self.left_bottom_ax_y, = self.bottom_ax.plot(left_profile_y, 'r', alpha=0.5, label='left plot')
        self.right_bottom_ax_y, = self.bottom_ax.plot(right_profile_y, 'b', alpha=0.5, label='right plot')
        self.bottom_ax.legend()
        self.top_ax.legend()

        # Plot reference image
        self.right_aximshow = self.right_img_ax.imshow(self.right_plot_img)
        self.right_img_ax.set_title('Right plot')

    def write_file(self, event):
        raise NotImplementedError

    def process_array(self, img, side):
        raise NotImplementedError

    @staticmethod
    def _get_varray(y):
        return [np.min(y), np.max(y)]

    def update(self, value):

        # Update slider values
        self.slider_value = int(self.slider_ax.val)
        self.r_slider_value = int(self.r_slider_ax.val)
        self.loca_value = int(self.slider_loca.val)
        self.rw_value = int(self.slider_rw.val)

        self.r_slider_ax.valtext.set_text('{}'.format(int(self.r_slider_ax.val)))
        self.slider_ax.valtext.set_text('{}'.format(int(self.slider_ax.val)))
        self.slider_loca.valtext.set_text('{}'.format(int(self.slider_loca.val)))
        self.slider_rw.valtext.set_text('{}'.format(int(self.slider_rw.val)))

        # Update image plot
        l_temp = self.left_plot_img[self.loca_value, :, self.slider_value:(self.slider_value+self.rw_value)]
        self.left_aximshow.set_data(l_temp)
        self.left_img_ax.set_xlim([self.slider_value, self.slider_value + self.rw_value])
        left_vmin = self._get_varray(l_temp)
        self.left_aximshow.set_clim(left_vmin)

        r_temp = self.right_plot_img[:, self.r_slider_value:(self.r_slider_value + self.rw_value)]
        self.right_aximshow.set_data(r_temp)
        self.right_img_ax.set_xlim([self.r_slider_value, self.r_slider_value + self.rw_value])
        right_vmin = self._get_varray(self.right_plot_img)
        self.right_aximshow.set_clim(right_vmin)

        # Update contour plot
        left_profile_x = np.sum(l_temp, axis=0)
        left_profile_y = np.sum(l_temp, axis=1)
        right_profile_x = np.sum(r_temp, axis=0)
        right_profile_y = np.sum(r_temp, axis=1)
        self.left_top_ax_x.set_data(range(len(left_profile_x)), left_profile_x)
        self.right_top_ax_x.set_data(range(len(right_profile_x)), right_profile_x)
        self.left_bottom_ax_y.set_data(range(len(left_profile_y)), left_profile_y)
        self.right_bottom_ax_y.set_data(range(len(right_profile_y)), right_profile_y)

        top_ax_max = np.ceil(np.max(np.concatenate([left_profile_x, right_profile_x])))
        bottom_ax_max = np.ceil(np.max([left_profile_y, right_profile_y]))
        self.top_ax.set_ylim([0, top_ax_max])
        self.bottom_ax.set_ylim([0, bottom_ax_max])

        # Draw data
        self.fig.canvas.draw()
        self.fig.canvas.draw()

    def flip(self, event):
        self.ind_flip = not self.ind_flip

        # Flip it!
        self.left_plot_img = np.flip(self.left_plot_img, axis=[-2, -1])

        self.update(None)

    def scale(self, event):
        if self.ind_scale:
            print('We have already scaled it')
        else:
            self.ind_scale = True
            left_max = np.amax(self.left_plot_img, axis=(-2, -1))
            self.left_plot_img = np.array([i_array / i_max for i_max, i_array in zip(left_max, self.left_plot_img)])
            self.right_plot_img = self.right_plot_img / np.max(self.right_plot_img)

            self.update(None)

    def align(self, event):
        res_list = []
        for slider_value in range(self.max_shift):
            l_temp = self.left_plot_img[self.loca_value, :, self.slider_value:(self.slider_value + self.rw_value)]
            r_temp = self.right_plot_img[:, self.r_slider_value:(self.r_slider_value + self.rw_value)]

            temp_x = np.sum(r_temp, axis=0)
            temp_y = np.sum(l_temp, axis=0)
            res = np.sum(temp_y * temp_x)
            res_list.append(res)

        arg_slider = np.argmax(res_list)
        self.slider_ax.val = arg_slider
        self.update(None)


class CompareB1Surv(ComparePlot):
    def __init__(self, x_obj, y_obj, dest_dir, file_prefix="19000101", train=True, **kwargs):
        super().__init__(x_obj, y_obj, dest_dir=dest_dir, file_prefix=file_prefix, train=train, **kwargs)

    def process_array(self, img, side='left'):
        # Here we process the left and right image.. that we get in..
        if side == 'left':
            # We only sum absolutely over the last eigh channels...
            left_plot_img = np.sum(np.abs(img[:, -8:]), axis=1)
            # Scale the left image with the height
            l_loc, l_h, l_w = left_plot_img.shape
            r_dyn, r_chan, r_h, r_w = self.right_img.shape
            c_factor = r_h / l_h
            l_h = int(c_factor * l_h)
            l_w = int(c_factor * l_w)
            l_shape = (l_loc, r_h, l_w)
            plot_img = sktransf.resize(left_plot_img, l_shape)
        elif side == 'right':
            plot_img = np.sum(np.sum(np.abs(img), axis=0), axis=0)
        else:
            raise ValueError

        return plot_img

    def write_file(self, event):

        if self.train:
            prefix_train = 'train'
        else:
            prefix_train = 'test'

        dest_dir_input = os.path.join(self.dest_dir, prefix_train, 'input')
        dest_dir_target = os.path.join(self.dest_dir, prefix_train, 'target')

        print('We want to write!')
        temp_left_img = self.left_img
        temp_right_img = self.right_img

        # Perform transformation
        if self.ind_flip:
            print('Perform flip')
            temp_left_img = np.flip(temp_left_img, axis=[-2, -1])

        # Selecting chosen location
        temp_left_img = np.take(temp_left_img, self.loca_value, axis=0)
        # Selecting channels
        n_chan, _, _ = temp_left_img.shape
        temp_left_img = np.take(temp_left_img, range(n_chan-8, n_chan), axis=0)

        # Store the data..
        for i, temp_coil in enumerate(temp_left_img):
            # Reshape the orignal data..
            temp_coil_real = sktransf.resize(np.real(temp_coil), self.left_plot_img[0].shape)
            temp_coil_imag = sktransf.resize(np.imag(temp_coil), self.left_plot_img[0].shape)
            temp_coil = temp_coil_real + 1j * temp_coil_imag

            temp_coil = temp_coil[:, self.slider_value:(self.slider_value + self.r_w)]
            # Normalize per coil image..
            if self.ind_scale:
                temp_coil /= np.max(temp_coil)

            temp_coil = np.stack([np.abs(temp_coil), np.angle(temp_coil)], axis=-1)
            file_name = self.file_prefix + '_{}.npy'.format(str(i))
            temp_path = os.path.join(dest_dir_input, file_name)
            print(temp_path)
            np.save(temp_path, temp_coil)

        for i, temp_dyn in enumerate(temp_right_img):
            if self.ind_scale:
                temp_dyn /= np.max(temp_dyn)

            temp_dyn = np.stack([np.abs(temp_dyn), np.angle(temp_dyn)], axis=-1)
            file_name = self.file_prefix + '_{}.npy'.format(str(i))
            temp_path = os.path.join(dest_dir_target, file_name)
            print(temp_path)
            np.save(temp_path, temp_dyn)


class PatchVisualizer:
    """An interactive polygon editor.
    Parameters
    ----------
    img_array: numpy array

    temp_y, temp_x = self.point_list[-1]
    """

    def __init__(self, img_array=None, patch_width=128, **kwargs):
        # Initial mask should be a polygon object
        self.debug = kwargs.get('debug')

        self.patch_width = patch_width
        self.img_array = img_array

        self.pos_x = None
        self.pos_y = None

        self.point_list = []
        self.point_ax_list = []

        if self.img_array is not None:
            self.img_shape = img_array.shape

            fig, ax = plt.subplots(1, 2)
            canvas = fig.canvas
            self.main_ax = ax[0]
            self.main_ax.imshow(self.img_array)

            self.second_ax = ax[1]
            self.patch_img = np.zeros((patch_width, patch_width))  # Dummy start..
            self.second_ax_imshow = self.second_ax.imshow(self.patch_img)

            # Callback definition
            self.canvas = canvas
            # canvas.mpl_connect('key_press_event', self.key_press_callback)
            # canvas.mpl_connect('button_press_event', self.button_press_callback)
            canvas.mpl_connect('button_release_event', self.button_release_callback)
            # canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        # Check that we are in the main ax..
        if event.inaxes == self.main_ax:
            if event.button == MouseButton.RIGHT:
                self.remove_point()
            elif event.button == MouseButton.LEFT:
                # Here we should plot a point
                pos_x = int(event.xdata)
                pos_y = int(event.ydata)
                self.plot_patch(pos_x=pos_x, pos_y=pos_y)
                self.plot_point(pos_x=pos_x, pos_y=pos_y)

        self.canvas.draw()

    def remove_point(self):
        # DIt kan beter.....
        if len(self.point_list) > 1:
            self.point_list.pop()
            temp_ax = self.point_ax_list.pop()
            temp_ax.remove()
            temp_y, temp_x = self.point_list[-1]
            self.plot_patch(pos_x=temp_x, pos_y=temp_y)
        elif len(self.point_list) == 1:
            self.point_list.pop()
            temp_ax = self.point_ax_list.pop()
            temp_ax.remove()

            temp_patch = np.zeros((self.patch_width, self.patch_width))  # Dummy start..
            temp_varray = (0, 1)
            self.second_ax_imshow.set_data(temp_patch)
            self.second_ax_imshow.set_clim(temp_varray)

    def get_patch(self, pos_x, pos_y, img=None):
        if img is None:
            img = self.img_array

        img_shape = img.shape

        min_x = max(0, pos_x - self.patch_width // 2)
        max_x = min(img_shape[1], pos_x + self.patch_width // 2)

        min_y = max(0, pos_y - self.patch_width // 2)
        max_y = min(img_shape[0], pos_y + self.patch_width // 2)

        patch_img = img[min_y: max_y, min_x: max_x]
        return patch_img

    def plot_patch(self, pos_x, pos_y):
        patch_img = self.get_patch(pos_x, pos_y)
        patch_varray = (patch_img.min(), patch_img.max())
        self.second_ax_imshow.set_data(patch_img)
        self.second_ax_imshow.set_clim(patch_varray)

    def plot_point(self,  pos_x, pos_y):
        point_ax = self.main_ax.scatter(pos_x, pos_y, color='k')
        self.point_ax_list.append(point_ax)
        self.point_list.append((pos_y, pos_x))

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


class RegionGrower:
    """
    This thing is not OK yet...
    Lets just create Edge Masks
    """
    def __init__(self, img, mask, thresh_width, p=1, fill_value=1):
        self.img = img
        self.mask = mask
        self.seed_list = self.create_seeds(mask)
        self.initial_mean_value = np.mean(img[mask==1])
        self.init_std = np.std(img[mask == 1])
        self.label = fill_value
        self.thresh_width = thresh_width
        self.connect_locations = self.select_connects(p)
        self.height, self.width = self.img.shape

    def create_seeds(self, mask):
        seeds = [Point(x, y) for x, y in np.argwhere(mask)]
        return seeds

    def get_gray_diff(self, mean_value, tmpPoint):
        # return abs(int(self.img[currentPoint.x, currentPoint.y]) - int(self.img[tmpPoint.x, tmpPoint.y]))
        return abs(mean_value - self.img[tmpPoint.x, tmpPoint.y])

    @staticmethod
    def select_connects(p):
        if p == 1:
            connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                        Point(0, 1), Point(-1, 1), Point(-1, 0)]
        else:
            connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
        return connects

    def update_values(self, cur_point, seed_mask):
        # Calculate a new mean value based on the neighbourhood...
        # combined_mask = seed_mask #* self.mask
        xmin = max(0, cur_point.x - self.thresh_width//2)
        xmax = min(self.height, cur_point.x + self.thresh_width//2)
        ymin = max(0, cur_point.y - self.thresh_width//2)
        ymax = min(self.width, cur_point.y + self.thresh_width//2)

        cropped_mask = seed_mask[xmin:xmax, ymin:ymax]
        cropped_img = self.img[xmin:xmax, ymin:ymax]
        if np.sum(cropped_mask == 1) > 0:
            mean_value = np.mean(cropped_img[cropped_mask == 1])
            std_value = np.std(cropped_img[cropped_mask == 1])
        else:
            mean_value = self.initial_mean_value
            std_value = self.init_std

        return mean_value, std_value

    def region_grow(self):
        seedMark = np.copy(self.mask)
        while len(self.seed_list) > 0:
            currentPoint = self.seed_list.pop(0)
            seedMark[currentPoint.x, currentPoint.y] = self.label
            for i in range(len(self.connect_locations)):
                tmpX = currentPoint.x + self.connect_locations[i].x
                tmpY = currentPoint.y + self.connect_locations[i].y
                tmpX_2 = currentPoint.x + 2 * self.connect_locations[i].x
                tmpY_2 = currentPoint.y + 2 * self.connect_locations[i].y
                if tmpX_2 < 0 or tmpY_2 < 0 or tmpX_2 >= self.height or tmpY_2 >= self.width:
                    continue
                if seedMark[tmpX, tmpY] == 0:
                    current_mean_value, current_std = self.update_values(currentPoint, seedMark)
                    grayDiff = self.get_gray_diff(current_mean_value, Point(tmpX, tmpY))
                    if grayDiff < current_std:
                        seedMark[tmpX, tmpY] = self.label
                        self.seed_list.append(Point(tmpX, tmpY))
        seedMark = harray.smooth_image(seedMark, n_kernel=3) > 0
        return seedMark


class ImageIntensityEqualizer:
    # Could also add the option to supply self created rcrop coords
    # --> It is adviced to give images in the range of 0..1...
    def __init__(self, reference_image, image_list, patch_width=None, dynamic_thresholding=False,
                 distance_measure='ssim'):
        # distance_measure - can be ssim or l2
        if patch_width is None:
            patch_width = min(reference_image.shape) // 4
        # Referenc image should be 2D
        # image_list should be list of 2D images...
        self.image_list = image_list
        self.ref_image = reference_image

        self.patch_width = patch_width  # Used for scaling..
        # Crop coords should be a list of coordinates.
        self.crop_coords = [harray.get_crop_coords_center(list(reference_image.shape), width=patch_width)]
        self.n_patches = len(self.crop_coords)
        self.n_images = len(image_list)

        self.dynamic_thresholding = dynamic_thresholding
        self.patches_image_list = self.get_patches_image_list(self.crop_coords)
        self.patches_ref = [harray.apply_crop(self.ref_image, i_coords) for i_coords in self.crop_coords]
        if distance_measure == 'ssim':
            self.distance_measure = hmisc.patch_min_fun_ssim
        elif distance_measure == 'l2':
            self.distance_measure = hmisc.patch_min_fun_l2
        else:
            print('Unknown distance measure ', distance_measure)
            print('Please choose ssim or l2')
            sys.exit()

        self.vmax_ref = self.get_vmax_ref_image()

    def plot_crop_coords(self, ax, sel_index=0):
        nx, ny = self.ref_image.shape[-2:]
        crop_coords = self.crop_coords[sel_index]
        ax.hlines(crop_coords[3], 0, ny - 1)
        ax.hlines(crop_coords[2], 0, ny - 1)
        ax.vlines(crop_coords[1], 0, nx - 1)
        ax.vlines(crop_coords[0], 0, nx - 1)
        return ax

    def measure_improvement(self, corrected_images):
        patch_corrected_images = self.get_patches_image_list(image_list=corrected_images)
        difference_uncor = np.round(np.linalg.norm(self.patches_ref - np.array(self.patches_image_list), axis=(-1, -2)), 2)
        difference_cor = np.round(np.linalg.norm(self.patches_ref - np.array(patch_corrected_images), axis=(-1, -2)), 2)
        print('Average difference when uncorrected ', np.mean(difference_uncor).round(2))
        print('Average difference when corrected ', np.mean(difference_cor).round(2))
        print('Ratio ', np.mean(difference_cor).round(2) / np.mean(difference_uncor).round(2))
        return np.mean(difference_cor).round(2) / np.mean(difference_uncor).round(2)

    def get_vmax_ref_image(self):
        # This is used as the cut-off value for this image
        patch_shape = tuple(np.array(self.ref_image.shape) // 10)
        stride = patch_shape[0] // 2
        vmax = harray.get_proper_scaled_v2(self.ref_image, patch_shape, stride)
        return vmax

    def get_patches_image_list(self, crop_coords=None, image_list=None):
        if image_list is None:
            image_list = self.image_list
        if crop_coords is None:
            crop_coords = self.crop_coords

        all_patches = []
        for i_img in image_list:
            # Double list is necessary for the rect of the code.
            temp = []
            for i_coord in crop_coords:
                temp_patch = harray.apply_crop(i_img, i_coord)
                temp.append(temp_patch)
            all_patches.append(temp)
        return all_patches

    def get_mean_scaling_from_patches(self):
        temp_scale_list = []
        for i in range(self.n_images):
            temp_scale = []
            for ii in range(self.n_patches):
                res = scipy.optimize.differential_evolution(self.distance_measure, strategy='randtobest1exp',
                                                            bounds=[(0, 20)],
                                                            args=(self.patches_image_list[i][ii], self.patches_ref[ii]),
                                                            x0=1)
                temp_scale.append(res.x[0])
            temp_scale_list.append(temp_scale)

        mean_scaling = np.mean(temp_scale_list, axis=(-1))
        return mean_scaling

    def apply_mean_scaling(self, mean_scaling):
        corrected_image_list = []
        for ii, i_img in enumerate(self.image_list):
            temp_img = i_img * mean_scaling[ii]
            corrected_image_list.append(temp_img)
        return corrected_image_list

    def apply_vmax_ref(self, image_list):

        for ii in range(self.n_images):
            temp_vmax_value = self.vmax_ref
            vmax_indices = image_list[ii] > temp_vmax_value
            # If we want to make sure that we dont get an image TOO bright, use dynamic tresholding
            if self.dynamic_thresholding:
                n_pixels = np.prod(image_list[ii].shape[-2:])
                cur_max_values_perc = np.sum(vmax_indices) / n_pixels
                # If the number of tresholded values takes more space than 1% of the whole image
                # We need to reduce vmax
                while cur_max_values_perc > 0.01:
                    vmax_indices = image_list[ii] > temp_vmax_value * 1.01
                    cur_max_values_perc = np.sum(vmax_indices) / n_pixels
                    temp_vmax_value = temp_vmax_value * 1.01
            # Replace all the values that are above the vmax-ref with the vmax-value.
            image_list[ii][vmax_indices] = temp_vmax_value
        return image_list

    def correct_image_list(self):
        # First set a scale, then a specific cut-off value.
        mean_scaling = self.get_mean_scaling_from_patches()
        # This below is also possible. But I believe it is less robust...
        # mean_scaling = np.array([np.mean(x) for x in self.patches_ref]) / np.array([np.mean(x) for x in self.patches_image_list])
        # Apply the found mean-scaling
        corrected_image_list = self.apply_mean_scaling(mean_scaling)
        # Apply the vmax ref now that the images have been scaled
        corrected_image_list = self.apply_vmax_ref(corrected_image_list)
        # Re-scale to an interval of 0..1 since vmax_ref might not be 1
        corrected_image_list = [harray.scale_minmax(x) for x in corrected_image_list]
        return corrected_image_list


class PlotCollage:
    def __init__(self, content_list, ddest, n_display, plot_type='file', subtitle_list=None,
                 proper_scaling=False, **kwargs):
        if subtitle_list is None:
            self.subtitle_list = [''] * n_display
        else:
            self.subtitle_list = subtitle_list

        self.height_offset = kwargs.get('height_offset', 0.05)
        self.plot_text_box = kwargs.get('text_box', True)
        self.fontsize = kwargs.get('fontsize', 16)
        self.dpi = kwargs.get('dpi', 300)
        self.cmap = kwargs.get('cmap', 'gray')
        self.sub_col_row = kwargs.get('sub_col_row', None)
        if self.sub_col_row is None:
            self.sub_col_row = hmisc.get_square(n_display)
        self.fig_size = list(np.array(self.sub_col_row) * 5)
        self.content_list = content_list
        self.plot_type = plot_type
        self.ddest = ddest
        self.n_display = n_display
        # This is used to forget the 'collage _filename_filename' string that is used to name the deitionat file
        self.only_str = kwargs.get('only_str', False)
        self.proper_scaling = proper_scaling
        self.proper_scaling_patch_shape = kwargs.get('proper_scaling_patch_shape', 128)
        self.proper_scaling_stride = kwargs.get('proper_scaling_stride', 64)

    def _get_file(self, ii):
        if self.plot_type == 'file':
            file_png = self.content_list[ii]
            base_name = hmisc.get_base_name(file_png)
            ext = hmisc.get_ext(file_png)
            plot_array = hmisc.load_array(file_png)
            if 'nii' in ext:
                plot_array = plot_array.T[0, ::-1, ::-1]
        else:
            base_name = f"array{str(ii).zfill(2)}"
            plot_array = self.content_list[ii]

        return plot_array, base_name

    def plot_collage(self, str_appendix=''):
        # Similar to the previous one.. just uses a list of arrays instead of files...
        n_files = len(self.content_list)
        multp = n_files // self.n_display
        n_files_multp = multp * self.n_display
        # Cut the number of images into groups of n_display
        for kk, index_range in enumerate(np.split(np.arange(n_files_multp), n_files_multp // self.n_display)):
            print(f'Processing file nr {kk}')
            plot_array = []
            file_string = ''
            for ii in index_range:
                temp_array, base_name = self._get_file(ii)
                plot_array.append(temp_array)
                # To prevent names that are too long..? Not sure about the empty _ being added
                if ii == index_range[0] or ii == index_range[-1]:
                    file_string += base_name + "_"
                else:
                    file_string += "_"

            if self.only_str:
                dest_file_name = os.path.join(self.ddest, str_appendix + '.png')
            else:
                dest_file_name = os.path.join(self.ddest, 'collage_' + file_string[:-1] + str_appendix + '.png')
            # plot_array = np.stack(plot_array)
            # print('Shape of plot array', plot_array.shape)
            # The height offset used here is obtained by simply trying.. It seems to work OK...
            temp_height = 2*self.sub_col_row[1]*self.height_offset
            fig_obj = ListPlot([plot_array], cmap=self.cmap, debug=False, subtitle=[self.subtitle_list],
                               col_row=(1, 1),
                               sub_col_row=self.sub_col_row, ax_off=True, wspace=0, hspace=temp_height,
                               figsize=self.fig_size, aspect='auto', proper_scaling=self.proper_scaling,
                               proper_scaling_patch_shape=self.proper_scaling_patch_shape, proper_scaling_stride=self.proper_scaling_stride)
            if self.plot_text_box:
                for jj, i_subtitle in enumerate(self.subtitle_list[:self.n_display]):
                    # Try to avoid circular imports..?
                    import helper.plot_fun as hplotf
                    hplotf.add_text_box(fig_obj.figure, jj, str(i_subtitle), height_rect=self.height_offset,
                                        linewidth=1, position='top', fontsize=self.fontsize)
            print('Storing to ', dest_file_name)
            fig_obj.figure.savefig(dest_file_name, bbox_inches='tight', pad_inches=0.0, dpi=self.dpi)
            close_all()

# Added this one so that I dont need to keep importing matplotlib..
def close_all():
    plt.close('all')

