import sys

import numpy as np

from PyQt5 import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import animation


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        central_widget = QtWidgets.QWidget()

        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        line_edit = QtWidgets.QLineEdit()
        line_edit.setMaxLength(2)
        line_edit.setValidator(QtGui.QIntValidator())
        line_edit.editingFinished.connect(self.test)
        layout.addWidget(line_edit)
        # layout.addWidget(self.animation_button)

    def test(self):
        print('hooi')
    def callback_animation(self, i):
        x, y = self.generate_data(i)
        self._line.set_data(x, y)
        # if self.animation_button.isChecked():
        #     self.writer.grab_frame()
        return (self._line,)

    def generate_data(self, i):
        x = np.linspace(0, 2, 1000)
        y = np.sin(2 * np.pi * (x - 0.01 * i))

        return x, y

    # def handle_toggled(self):
    #     self.animation_button.setText(
    #         "Stop" if self.animation_button.isChecked() else "Start"
    #     )

    # def closeEvent(self, event):
    #     super().closeEvent(event)
    #     self.writer.finish()


if __name__ == "__main__":
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec_()

# for line in lines:
#         f.write(f"{line}\n")

#
import sys
sys.path.append(r'F:\code_backup\pytorch_in_mri')
import itertools
import helper.misc as hmisc
import numpy as np
import collections
import re
import importlib
import pandas as pd
import datetime

import os
import data_prep.dataset.cardiac.scan_7T.helper_scan_7T as helper_scan_7T
import matplotlib.pyplot as plt

"""
I think I need to do this...
Push them all through the same pipeline. Because the .par/.cpx files already have that black cirlce around the radial files...

"""

ddest = '/home/bugger/Documents/paper/undersampled_recon'
ddest_data = r'F:\data\data\7T_data\cardiac_cine_mat'

# Read both cartesian and radial database
radial_database = pd.read_csv(os.path.join(ddest, 'scan_radial_files.csv'))
cartesian_database = pd.read_csv(os.path.join(ddest, 'scan_cartesian_files.csv'))

os.listdir(ddest_data)
# Also read the files that we currently have processed
dataframe_scan = helper_scan_7T.get_data_frame_scan_files(ddest_data)
