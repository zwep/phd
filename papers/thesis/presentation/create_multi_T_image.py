import os
import numpy as np
import harreveltools.plot as hplot


ddata = '/media/bugger/MyBook/data/multiT_scan/'
subject_fieldstrength = {'1.5T': [], '3T': [], '7T': []}

for i_sub in subject_fieldstrength.keys():
    field_strength_sub = os.path.join(ddata, i_sub)
    subject_ID = [x for x in os.listdir(field_strength_sub) if x.startswith('V')]
    subject_fieldstrength[i_sub] = set(subject_ID)

# This was used to find overlapping Vnumbers
overlap = set.intersection(*[v for k, v in subject_fieldstrength.items()])

# Now we are going to create the images... (or load them)
import pydicom
ddata_1p5T = '/media/bugger/MyBook/data/multiT_scan/1.5T/V7908/DICOM/IM_0001'
ddata_3T = '/media/bugger/MyBook/data/multiT_scan/3T/V7753/DICOM/IM_0013'
ddata_7T = '/media/bugger/MyBook/data/multiT_scan/7T/V7908/DICOM/IM_0008'

import harreveltools.file_handling as hfile
import harreveltools.data_transform as htransf
import helper.array_transf as harray

img_1pT5 = hfile.load_array(ddata_1p5T)
img_3T = hfile.load_array(ddata_3T)
img_3T = img_3T[:, ::-1]
img_7T = hfile.load_array(ddata_7T)

plot_list = [img_1pT5, img_3T[0], img_7T[0]]
plot_list = [htransf.scale_minmax(x, axis=(-2, -1)) for x in plot_list]
patch_size = min([min(x.shape) for x in plot_list]) // 10
vmax_list = [(0, harray.get_proper_scaled_v2(x, (patch_size, patch_size), patch_size // 2)) for x in plot_list]

"""         Plot the plot array       """
fig_obj = hplot.ListPlot(plot_list, ax_off=True, hspace=0, wspace=0, vmin=vmax_list, figsize=(30, 10), dpi=600, col_row=(3,1))
# fig_obj = hplotc.ListPlot(plot_array[None], ax_off=True, hspace=0, wspace=0,  figsize=(30, 10))


hplot.ListPlot([img_1pT5, img_3T[0], img_7T[0]], ax_off=True, col_row=(3, 1))
