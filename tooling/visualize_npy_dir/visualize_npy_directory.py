import json
import itertools
from matplotlib.backend_bases import MouseButton

import numpy as np
import sys

import PyQt5.QtGui
import PyQt5.QtCore
import PyQt5.QtWidgets

import matplotlib.pyplot as plt

import tooling.genericinterface as gen_interface
import os
import helper.misc as hmisc

"""
Bleeh visualize stuff..
Create interactive graphs... when clicked, store that name + slice + axis somewhere
Later, make a count of the 'best' predictions

Visualization is similar to what you have made for the paper
"""


class PickBestImage(gen_interface.GenericInterface):
    def __init__(self, ddir, subdir=None, reload_counter_dict=False, file_index=None):
        super().__init__()

        # This is the main path..
        self.ddir = ddir
        if subdir is None:
            self.sub_dir = os.listdir(self.ddir)
        else:
            self.sub_dir = subdir
        n_axes = len(self.sub_dir) + 2
        temp_path = os.path.join(self.ddir, self.sub_dir[0])
        self.file_list_pred = sorted([x for x in os.listdir(temp_path) if x.startswith('pred')])
        self.file_list_target = sorted([x for x in os.listdir(temp_path) if x.startswith('target')])
        self.file_list_input = sorted([x for x in os.listdir(temp_path) if x.startswith('input')])
        self.n_files = len(self.file_list_pred)

        self.display_dict = {k: {} for k in self.sub_dir}
        if reload_counter_dict:
            temp_config_name = os.path.join('/media/bugger/MyBook/data/7T_data/prostate_semireal_data/counter_dict')
            with open(temp_config_name, 'r') as f:
                temp_str = f.read()
            self.counter_dict = json.loads(temp_str)
            print('Loaded the current dict')
            hmisc.print_dict(self.counter_dict)
        else:
            self.counter_dict = {k: [] for k in ['none'] + self.sub_dir}


        self.target_array = None
        self.input_array = None
        self.press_indicator = False
        self.press_position = None
        self.epsilon = 0.001

        if file_index is None:
            self.file_index = 0
        else:
            self.file_index = file_index

        i_row, i_col = hmisc.get_square(n_axes)
        self.canvas, self.axes = self.get_canvas(i_row, i_col)
        self.axes = list(self.axes.ravel())
        self.axes_imshow = []

        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('motion_notify_event', self.move_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)

        h_layout = PyQt5.QtWidgets.QHBoxLayout(self._main)
        h_layout.addWidget(self.canvas)

    def button_press_callback(self, event):
        current_file = self.file_list_pred[self.file_index]
        if event.inaxes in self.axes:
            index_value = self.axes.index(event.inaxes)
            if event.button == MouseButton.RIGHT:
                if index_value == 0:
                    print('You have selected the target you dummy')
                elif index_value == 1:
                    self.counter_dict['none'].append(current_file)
                    print('One up for.. nobody')
                else:
                    # This is 2 because we added the target/input image on position 0 and 1
                    self.counter_dict[self.sub_dir[index_value - 2]].append(current_file)
                    print('One up for..', self.sub_dir[index_value - 2])

            elif event.button == MouseButton.MIDDLE:
                if index_value == 0:
                    print('You have selected the target you dummy')
                elif index_value == 1:
                    self.counter_dict['none'].pop()
                    print('One down for.. nobody')
                else:
                    # This is 2 because we added the target/input image on position 0 and 1
                    self.counter_dict[self.sub_dir[index_value - 2]].pop()
                    print('One down for..', self.sub_dir[index_value - 2])

            elif event.button == MouseButton.LEFT:
                self.press_indicator = True
                self.press_position = {'x': event.xdata, 'y': event.ydata}
                self.press_clim = list(self.axes_imshow[index_value].get_clim())

    def move_callback(self, event):
        if event.inaxes in self.axes:
            if self.press_indicator:
                current_position = {'x': event.xdata, 'y': event.ydata}
                index_current_ax = self.axes.index(event.inaxes)

                size_x, size_y = self.axes_imshow[index_current_ax].get_size()
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
                self.axes_imshow[index_current_ax].set_clim(new_clim)
                self.canvas.draw()

    def button_release_callback(self, event):
        # if event.inaxes:
        self.press_indicator = False
        self.press_position = None

    def change_file(self, mod=0):
        self.file_index += mod
        self.file_index = self.file_index % self.n_files
        self.load_file()
        for i_ax in self.axes:
            i_ax.cla()

        self.display_loaded_data()
        for k, v in self.counter_dict.items():
            print(k)
            print('Count ', len(v))

        ser_json_config = json.dumps(self.counter_dict)
        temp_config_name = os.path.join('/media/bugger/MyBook/data/7T_data/prostate_semireal_data/counter_dict')
        with open(temp_config_name, 'w') as f:
            f.write(ser_json_config)

    def change_slice(self, mod=0):
        pass

    def display_loaded_data(self):
        print('Display stuff')
        self.axes_imshow = []
        for i_ax, i_sub in zip(self.axes, ['target', 'input'] + self.sub_dir):
            if i_sub == 'target':
                temp_ax = i_ax.imshow(self.target_array, cmap='gray')
                i_ax.set_title('target')
            elif i_sub == 'input':
                temp_ax = i_ax.imshow(self.input_array, cmap='gray')
                i_ax.set_title('input')
            else:
                temp_ax = i_ax.imshow(self.display_dict[i_sub]['prediction'], cmap='gray')
                i_ax.set_title(i_sub)

            self.axes_imshow.append(temp_ax)

        self.canvas.figure.suptitle(f'File {self.file_index} / {self.n_files}')
        self.canvas.draw()

    def load_file(self):
        for i_sub in self.sub_dir:
            load_file = os.path.join(ddata, i_sub, self.file_list_pred[self.file_index])
            print('Loading file ', load_file)
            temp_A = np.load(load_file)
            self.display_dict[i_sub]['prediction'] = temp_A

        target_load_file = os.path.join(ddata, i_sub, self.file_list_target[self.file_index])
        self.target_array = np.load(target_load_file)
        input_load_file = os.path.join(ddata, i_sub, self.file_list_input[self.file_index])
        self.input_array = np.load(input_load_file)


if __name__ == '__main__':
    ddata = '/media/bugger/MyBook/data/7T_data/prostate_semireal_data/test_split_results'
    # Each one is usefull.....

    qapp = PyQt5.QtWidgets.QApplication(sys.argv)
    subdir = [x for x in os.listdir(ddata) if 'one_slice' not in x and 'exp' not in x]
    app = PickBestImage(ddir=ddata, subdir=subdir, reload_counter_dict=False, file_index=382)
    app.show()
    qapp.exec_()
