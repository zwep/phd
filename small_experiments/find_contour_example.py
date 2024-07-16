
import os
import helper.array_transf as harray
import helper.plot_class as hplotc
import numpy as np
from PIL import Image

"""
Create masks based on the jpeg images....

We found that the creation of the mask is not perfect.
Editing with our mask tool became too annoying. So we took a different route:
Create a mask on the whole refscan image by extrapolation, then slice that. 
"""

ddata = '/media/bugger/MyBook/data/7T_data/cardiac/transverse'


ddata_files = [os.path.join(ddata, x) for x in os.listdir(ddata)]

sel_file = ddata_files[0]
A = np.array(Image.open(sel_file))
A_mask = harray.get_treshold_label_mask(A)
hplotc.ListPlot([A, A_mask])

import cv2
import matplotlib.pyplot as plt
contours, hier = cv2.findContours(A_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
x_coord = contours[0][:, 0, 0]
y_coord = contours[0][:, 0, 1]
plt.scatter(x_coord, y_coord)
from matplotlib.patches import Polygon
poly = Polygon(np.array([x_coord, y_coord]).T, animated=True, fc='y', ec='none', alpha=0.4)
line = plt.Line2D([x_coord], [y_coord], color='black', marker='o', mfc='r', alpha=0.8, animated=True)

import importlib
importlib.reload(hplotc)
mask_obj = hplotc.MaskCreator(A[None], initial_mask=poly, debug=True)
mask_obj.main_ax.scatter(100, 100, s=100)
mask_obj.main_ax.scatter(0, 0, s=100)
mask_obj.main_ax.scatter(600, 600, s=100)
poly.axes = mask_obj.main_ax
poly.figure = mask_obj.main_ax.figure
mask_obj.main_ax.draw_artist(poly)
