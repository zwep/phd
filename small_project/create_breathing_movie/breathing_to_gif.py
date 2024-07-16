import os
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from tooling.genericinterface import GenericInterface
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import PyQt5.QtGui
import PyQt5.QtCore
import PyQt5.QtWidgets


class BreathingAnimation(PyQt5.QtWidgets.QMainWindow):
    # Could inherit from GenericInterface
    # Animate three ways:
    # o Fixed sinusoid -> moving ball
    # o Fixed ball (up/down) -> moving wave
    # o Expanding ball

    def __init__(self, period=1, delta_t=0.01, animation_type='ball'):
        super().__init__()
        self._main = PyQt5.QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        """
        Plot variables
        
        sin(omega * x)
        
        period = 2 * np.pi * omega
        omega = period / (2 * np.pi)
        """
        self.delta_t = delta_t
        self.frequentie = 1/period
        self.max_plot_time = 5
        self.max_periods = 20
        self.t_range = np.arange(0, self.max_periods * period, delta_t)
        self.animation_type = animation_type
        self.cool_down = 10
        self.prev_time = time.time()
        self.prev_time_cor_time = time.time()
        """
        Graphical definitions
        """
        self.canvas, self.fig, self.axes = self.get_figure(self.max_plot_time)
        self.axes.set_axis_off()
        self.line_obj, self.scatter_obj = self.get_plot_objects(self.axes)

        # Get the animation object
        self.anim_obj = self.get_animation()

        h_layout = PyQt5.QtWidgets.QHBoxLayout(self._main)
        # Create buttons
        # self.button_box_layout = self.get_button_box_layout()
        # write_button = self.get_push_button(name='Write', shortcut='W', connect=self.write_animation)
        # self.edit_button = self.get_line_edit(name=f'{period}', connect=self.update_period, max_length=4)
        # self.button_box_layout.addWidget(self.edit_button)
        # self.button_box_layout.addWidget(write_button)

        # Add canvas to the figure
        temp_canvas_layout = PyQt5.QtWidgets.QVBoxLayout()
        temp_canvas_layout.addWidget(self.canvas)

        h_layout.addLayout(temp_canvas_layout, stretch=1)
        # h_layout.addLayout(self.button_box_layout, stretch=0.01)

    @staticmethod
    def get_figure(max_plot_time, debug=False):
        if debug:
            fig = plt.figure()
        else:
            fig = Figure(figsize=(5, 5), dpi=100)
        canvas = FigureCanvas(fig)
        axes = canvas.figure.subplots()
        # self.axes.set_axis_off()
        axes.set_ylim(-2, 2)
        axes.set_xlim(0, max_plot_time)
        return canvas, fig, axes

    @staticmethod
    def get_plot_objects(axes):
        # Create a line object
        line_obj = axes.plot([], [], zorder=1)[0]
        # Create a scatter object
        scatter_obj = axes.scatter([], [], s=40, marker='o', c='r')
        return line_obj, scatter_obj

    def get_y_value(self, i_t):
        omega = 2 * np.pi * self.frequentie
        y_value = np.sin(omega * i_t)
        return y_value

    def animate_moving_ball(self, i, line_obj=None, scatter_obj=None):
        i = i % len(self.t_range)
        if line_obj is None:
            line_obj = self.line_obj
        if scatter_obj is None:
            scatter_obj = self.scatter_obj
        line_obj.set_data(self.t_range, self.get_y_value(self.t_range))
        sel_time = self.t_range[i]
        scatter_obj.set_offsets(np.c_[sel_time, self.get_y_value(sel_time)])
        return scatter_obj

    def animate_moving_wave(self, i, line_obj=None, scatter_obj=None):
        i = i % len(self.t_range)
        if line_obj is None:
            line_obj = self.line_obj
        if scatter_obj is None:
            scatter_obj = self.scatter_obj
        line_obj.set_data(self.t_range, np.roll(self.get_y_value(self.t_range), -i))
        sel_time = self.t_range[i] + self.max_plot_time/2.
        # print(f'max plot time {i}', self.max_plot_time/2, self.get_y_value(sel_time))
        scatter_obj.set_offsets(np.c_[self.max_plot_time/2., self.get_y_value(sel_time)])
        # self.cool_down -= 1
        # # print(self.cool_down)
        if ((1 - self.get_y_value(sel_time)) < 0.0001):
            time_difference = time.time() - self.prev_time
            self.prev_time = time.time()
            print('Time interval in seconds ', time_difference)

        return scatter_obj

    def update_period(self):
        new_periode = float(self.edit_button.text())
        self.frequentie = 1./new_periode
        self.t_range = np.arange(0, self.max_periods * new_periode, self.delta_t)

    def get_animation(self):
        if self.animation_type == 'ball':
            # Return a moving ball..
            animation_fun = self.animate_moving_ball
        elif self.animation_type == 'wave':
            # Return a wave..
            animation_fun = self.animate_moving_wave
        else:
            animation_fun = None

        self.animation_obj = animation.FuncAnimation(self.canvas.figure, animation_fun,
                                                     blit=False, repeat=True,
                                                     interval=self.delta_t,  # Delay in ms
                                                     frames=len(self.t_range))
        self.animation_obj.new_frame_seq()
        return self.animation_obj

    def write_animation(self):
        num_frames = len(self.t_range)
        max_time = np.max(self.t_range) # in seconds?
        print('frames ', num_frames / max_time)
        ffmpeg_writer = animation.FFMpegWriter(fps=num_frames / max_time)
        self.animation_obj.save(os.path.expanduser('~/breathing_animation.mp4'), writer=ffmpeg_writer)
        print('Written')


if __name__ == "__main__":
    qapp = PyQt5.QtWidgets.QApplication(sys.argv)
    app = BreathingAnimation(period=3, animation_type='wave', delta_t=0.009)
    app.show()
    qapp.exec_()
