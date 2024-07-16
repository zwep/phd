"""
Copied to local from legolas
"""

import helper.array_transf as harray
import os
import helper.plot_class as hplotc
import nibabel
import numpy as np

ddata = '/home/bugger/Documents/data/mm1/vendor_B/img'
ddata_label = '/home/bugger/Documents/data/mm1/vendor_B/label'

for i_file in os.listdir(ddata):
    ddata_file = os.path.join(ddata, i_file)
    ddata_file_label = os.path.join(ddata_label, i_file)

    loaded_array = nibabel.load(ddata_file).get_fdata()
    loaded_array_label = nibabel.load(ddata_file_label).get_fdata()
    sel_array = np.moveaxis(loaded_array, -1, 0)[0]
    hplotc.ListPlot([sel_array, harray.get_treshold_label_mask(sel_array, class_treshold=0.001, treshold_value=0.01)])
    # hplotc.SlidingPlot(np.moveaxis(loaded_array_label, -1, 0))
