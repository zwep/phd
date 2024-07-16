import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axes_grid1 import make_axes_locatable

import helper.array_transf as harray

import numpy as np


class B1ManualShimming:
    def __init__(self, img_array, slider_name=None, fignum=None, **kwargs):
        # Some initial parapmeters
        # Counts the amount of shim settings
        n_c, n_y, n_x = img_array.shape
        self.input_array = img_array
        self.plot_data = np.abs(img_array)
        self.varray = [np.min(self.plot_data), np.max(self.plot_data)]
        self.axcolor = 'lightgoldenrodyellow'
        self.delta_ax = 0.05
        self.n_slider = n_c
        self.min_slider = [-np.pi for x in range(n_c)]
        self.max_slider = [np.pi for x in range(n_c)]
        self.initial_angle = kwargs.get('initial_angle', np.pi/2)

        if slider_name is None:
            self.slider_name = ['Axes {}'.format(str(i)) for i in range(self.n_slider)]
        else:
            self.slider_name = slider_name

        self.fig, self.ax = plt.subplots(num=fignum)
        plt.subplots_adjust(left=0.25, top=0.8, bottom=0.2)
        divider = make_axes_locatable(self.ax)
        self.axcolorbar = divider.append_axes('right', size='5%', pad=0.05)

        # Create slider
        self.slider_value = np.array(np.zeros(self.n_slider))
        self.sliderax_list = [self.fig.add_axes([0.05, 0.25 + self.delta_ax * i, 0.1, 0.03], facecolor=self.axcolor) for
                              i in range(self.n_slider)]
        self.slider_list = [Slider(self.sliderax_list[i], self.slider_name[i],
                                   self.min_slider[i],
                                   self.max_slider[i],
                                   valinit=0) for i in range(self.n_slider)]
        [x.on_changed(self.update) for x in self.slider_list]

        x_opt = np.exp(1j * self.slider_value)
        self.plot_data = np.abs(np.einsum("tmn, t -> mn", self.input_array, x_opt))
        self.plot_data = harray.scale_minmax(self.plot_data) * self.initial_angle

        self.aximshow = self.ax.imshow(self.plot_data, vmin=self.varray)
        self.scalarmap = cm.ScalarMappable(norm=mcolors.Normalize(*self.varray))
        self.colorbarobj = self.fig.colorbar(self.scalarmap, ax=self.aximshow, cax=self.axcolorbar)

        self.rax = plt.axes([0.05, 0.02, 0.1, 0.15], facecolor=self.axcolor)
        self.transform_type = 'none'
        self.radio = RadioButtons(self.rax, ('none', 'sin3', 'sin7', 'sin11'), active=0)
        self.radio.on_clicked(self.set_transform_type)

        self.rescaleax = plt.axes([0.05, 0.2, 0.1, 0.03])
        self.button_rescale = Button(self.rescaleax, 'Resscale', color=self.axcolor, hovercolor='0.975')
        self.button_rescale.on_clicked(self.rescale)

    def rescale(self, event):
        # temp_data = self.plot_data[self.slider_value]
        # self.plot_data[self.slider_value] = temp_data / np.max(temp_data)
        # Update clim
        min_plot = np.min(self.plot_data)
        max_plot = np.max(self.plot_data)
        self.varray = [min_plot, max_plot]
        self.aximshow.set_clim(self.varray)
        self.scalarmap.set_clim(self.varray)
        self.update(None)

    def _aug_data(self, data):
        print('mean / max ', data.mean(), data.max())
        if self.transform_type == 'sin3':
            temp_data = np.sin(data) ** 3
        elif self.transform_type == 'sin11':
            temp_data = np.sin(data) ** 11
        elif self.transform_type == 'sin7':
            temp_data = np.sin(data) ** 7
        else:
            temp_data = data

        return temp_data

    def set_transform_type(self, label):
        print('Setting label ', label)
        self.transform_type = label
        self.update(None)

    def update(self, value):
        # unpack list and use that to subset data
        self.slider_value = np.array([float(x.val) for x in self.slider_list])
        [x.valtext.set_text('{:.2f}'.format(float(x.val))) for x in self.slider_list]
        # Update slider setting
        x_opt = np.exp(1j * self.slider_value)
        self.plot_data = np.abs(np.einsum("tmn, t -> mn", self.input_array, x_opt))
        # Scale to 0.. pi/2...
        self.plot_data = harray.scale_minmax(self.plot_data) * self.initial_angle
        self.plot_data = self._aug_data(self.plot_data)
        self.aximshow.set_data(self.plot_data)

        # Draw everything
        self.colorbarobj.draw_all()
        self.fig.canvas.draw()
