import matplotlib.pyplot as plt
import helper.plot_class as hplotc
import importlib

importlib.reload(hplotc)

import skimage.data as skdata
A = skdata.astronaut()[:, :, 0]
B = skdata.logo()[:, :, 0]
overal_list = []

plot_obj = hplotc.PatchVisualizer(A)

img_list = [plot_obj.img_array]
for ix, iy in plot_obj.point_list:
    temp_patch = plot_obj.get_patch(iy, ix)
    img_list.append(temp_patch)

overal_list.append(img_list)

import helper.misc as hmisc

hplotc.ListPlot(hmisc.change_list_order(overal_list), ax_off=True)
