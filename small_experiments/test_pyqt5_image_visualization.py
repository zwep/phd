# Base code from
# https://www.geeksforgeeks.org/pyqtgraph-setting-image-to-image-view/

"""
This is going nowhere...
"""


# importing Qt widgets
from PyQt5.QtWidgets import *

# importing system
import sys

# importing numpy as np
import numpy as np

# importing pyqtgraph as pg
import pyqtgraph as pg
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

A = np.random.rand(100, 100)

def window():
   win = QWidget()
   l1 = QLabel()
   self.photo.setGeometry(QtCore.QRect(10, 20, 761, 651))
   l1.setPixmap(QPixmap("/home/bugger/Pictures/Screenshot from 2021-08-29 22-26-42"))
   l2 = QLabel()
   l2.setPixmap(QPixmap("/home/bugger/Pictures/Screenshot from 2021-08-29 22-26-42"))

   vbox = QVBoxLayout()
   vbox.addWidget(l1)
   vbox.addWidget(l2)
   win.setLayout(vbox)
   win.setWindowTitle("QPixmap Demo")
   return win


if __name__ == "__main__":
   qapp = QApplication(sys.argv)
   app = window()
   app.show()
   qapp.exec_()