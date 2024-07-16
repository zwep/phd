

import numpy as np
import sys

import PyQt5.QtGui
import PyQt5.QtCore
import PyQt5.QtWidgets

import matplotlib.pyplot as plt
# encoding: utf-8

"""
Here we have created a tool in PyQt5 to allow for an interactive shimming session
"""

import PyQt5.QtGui
import PyQt5.QtCore
import PyQt5.QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class GenericInterface(PyQt5.QtWidgets.QMainWindow):
    """
    Generic interface. Creates the following attributes that should be re-used

    toolbar
    statusBar
    file_menu
    button_box_layout
    """
    def __init__(self, activate_status_bar=True, activate_tool_bar=True,
                 activate_menu=True,
                 **kwargs):
        super().__init__()

        self.debug = kwargs.get('debug')

        # ============== PyQt5 Impl ==============
        # Defining the Layout
        # This is really necessary for everything to work.
        self._main = PyQt5.QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        # Add Left side menu...
        self.button_box_layout = self.get_button_box_layout()

        # Add toolbar
        # These are the navigation buttons on the top
        if activate_tool_bar:
            toolbar = self.addToolBar('Navigate')
            prev_file, prev_slice, next_slice, next_file = self.get_toolbar_actions()
            toolbar.addAction(prev_file)
            toolbar.addAction(prev_slice)
            toolbar.addAction(next_slice)
            toolbar.addAction(next_file)

        # Create a status bar
        if activate_status_bar:
            self.statusBar()

        # Create a menu
        # These are the menu options such as File, ..., Help and such
        if activate_menu:
            file_menu = self.get_menu()
            exit_action = self.get_action(icon_name='exit.png',
                                          action_name='&Exit',
                                          shortcut='Ctrl+Q',
                                          connect=PyQt5.QtWidgets.qApp.quit,
                                          description='Exit application')

            open_action = self.get_action(icon_name='open.png',
                                          action_name='&Open',
                                          shortcut='Ctrl+O',
                                          connect=self.open_file,
                                          description='Open file browser')

            file_menu.addAction(exit_action)
            file_menu.addAction(open_action)

    def change_slice(self, mod=0):
        raise NotImplementedError

    def change_file(self, mod=0):
        raise NotImplementedError

    def open_file(self):
        raise NotImplementedError

    def get_toolbar_actions(self):
        prev_file_act = PyQt5.QtWidgets.QAction('Previous file', self)
        prev_file_act.setIcon(PyQt5.QtWidgets.QApplication.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_MediaSeekBackward))
        prev_file_act.setStatusTip('Previous file')
        prev_file_act.setShortcut('P')
        prev_file_act.triggered.connect(lambda: self.change_file(-1))

        prev_slice_act = PyQt5.QtWidgets.QAction('Previous slice', self)
        prev_slice_act.setIcon(PyQt5.QtWidgets.QApplication.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_ArrowBack))
        prev_slice_act.setStatusTip('Previous slice')
        prev_slice_act.triggered.connect(lambda: self.change_slice(-1))

        next_slice_act = PyQt5.QtWidgets.QAction('Next slice', self)
        next_slice_act.setIcon(PyQt5.QtWidgets.QApplication.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_ArrowForward))
        next_slice_act.setStatusTip('Next slice')
        next_slice_act.triggered.connect(lambda: self.change_slice(1))

        next_file_act = PyQt5.QtWidgets.QAction('Next file', self)
        next_file_act.setIcon(PyQt5.QtWidgets.QApplication.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_MediaSeekForward))
        next_file_act.setStatusTip('Next file')
        next_file_act.setShortcut('N')
        next_file_act.triggered.connect(lambda: self.change_file(1))

        return prev_file_act, prev_slice_act, next_slice_act, next_file_act

    def get_button_box_layout(self):
        button_box_layout = PyQt5.QtWidgets.QVBoxLayout()
        button_box_layout.setSpacing(5)
        button_box_layout.setAlignment(PyQt5.QtCore.Qt.AlignTop)

        return button_box_layout

    def get_line_edit(self, name=None, connect=None, max_length=2):
        line_edit = PyQt5.QtWidgets.QLineEdit()
        line_edit.setMaxLength(max_length)
        # line_edit.setValidator(PyQt5.QtGui.QIntValidator())
        if name is not None:
            line_edit.setText(name)
        if connect is not None:
            line_edit.editingFinished.connect(connect)
        return line_edit

    def get_push_button(self, name=None, shortcut=None, connect=None):
        push_button = PyQt5.QtWidgets.QPushButton()
        if name is not None:
            push_button.setText(name)
        if shortcut is not None:
            push_button.setShortcut(shortcut)
        if connect is not None:
            push_button.clicked.connect(connect)
        return push_button

    def get_canvas(self, nrows=1, ncols=1):
        # This should still be added to a Layout...
        # i.e. h_layout.addWidget(self.canvas)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(PyQt5.QtWidgets.QSizePolicy.Expanding,
                             PyQt5.QtWidgets.QSizePolicy.Expanding)
        canvas.updateGeometry()
        canvas.setFocusPolicy(PyQt5.QtCore.Qt.ClickFocus)
        return canvas, axes

    def get_slider(self, min=0, max=100, orientation='horizontal', connect=None):
        if orientation == 'horizontal':
            pyqt_orientation = PyQt5.QtCore.Qt.Horizontal
        elif orientation == 'vertical':
            pyqt_orientation = PyQt5.QtCore.Qt.Vertical
        else:
            pyqt_orientation = None
            print('Unkown orientation')

        slider = PyQt5.QtWidgets.QSlider(pyqt_orientation)
        slider.setMinimum(min)
        slider.setMaximum(max)
        slider.setTickInterval(1)
        slider.setSingleStep(1)

        if connect is not None:
            slider.valueChanged.connect(connect)
        return slider

    def get_dropdown(self, options, current_index, description, connect=None):
        dropwdown_obj = PyQt5.QtWidgets.QComboBox()
        dropwdown_obj.addItems(options)
        dropwdown_obj.setCurrentIndex(current_index)
        dropwdown_obj.setStatusTip(description)
        if connect is not None:
            dropwdown_obj.currentIndexChanged.connect(self.draw_callback)

        return dropwdown_obj

    def get_checkbox(self, name, state=False, shortcut=None, connect=None):
        check_box_obj = PyQt5.QtWidgets.QCheckBox(name, self)
        check_box_obj.setChecked(state)

        if shortcut is not None:
            check_box_obj.setShortcut(shortcut)

        if connect is not None:
            check_box_obj.stateChanged.connect(connect)

        return check_box_obj

    def get_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        return file_menu

    def get_action(self, icon_name, action_name, shortcut, description, connect=None):
        qt_action = PyQt5.QtWidgets.QAction(PyQt5.QtGui.QIcon(icon_name), action_name, self)
        qt_action.setShortcut(shortcut)
        qt_action.setStatusTip(description)
        if connect is not None:
            qt_action.triggered.connect(connect)

        return qt_action


if __name__ == '__main__':
    qapp = PyQt5.QtWidgets.QApplication(sys.argv)
    app = GenericInterface()
    app.show()
    qapp.exec_()
