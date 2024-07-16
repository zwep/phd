"""

Here I plot some stuff locally...

I
"""

from matplotlib.colors import ListedColormap
import json
import helper.misc as hmisc
import nibabel
import os
import data_generator.Segment7T3T as data_gen
import helper.plot_class as hplotc
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import ImageColor

class_interpretation = {'1': 'RV', '2': 'MYO', '3': 'LV'}
color_dict = {'1': '#1E88E5', '2': '#D81B60', '3': '#004D40'}
color_dict_rgb = {k: np.array(ImageColor.getcolor(x, "RGB"))/256 for k, x in color_dict.items()}

dvisual = '/home/bugger/Documents/presentaties/Espresso/januari_2022/for_performance_model_results'
dimages = '/home/bugger/Documents/presentaties/Espresso/januari_2022/7T_examples'
dlabels = '/home/bugger/Documents/presentaties/Espresso/januari_2022/7T_labels'
dnnunet = '/home/bugger/Documents/presentaties/Espresso/januari_2022/for_performance_model_results/nnunet_results'
nnunet_files = os.listdir(dnnunet)



for i_file in nnunet_files:
    base_name = hmisc.get_base_name(i_file)
    temp_file_path = os.path.join(dnnunet, i_file)
    orig_7T_file = os.path.join(dimages, i_file)
    nnunet_label = np.moveaxis(np.array(nibabel.load(temp_file_path).get_fdata()), -1, 0)
    image_7T = np.moveaxis(np.array(nibabel.load(orig_7T_file).get_fdata()), -1, 0)
    fig_obj = hplotc.ListPlot([(image_7T[10:20])], ax_off=True)
    # Here we can plot segmentations the nice way
    for i_ax, i_img in zip(fig_obj.ax_list, nnunet_label[10:20]):
        mask_values = i_img == 0
        cmap_all = np.array(list(color_dict_rgb.values()))
        my_cmap = ListedColormap(cmap_all)
        i_img = np.ma.masked_where(mask_values, i_img)
        i_ax.imshow(i_img, cmap=my_cmap, alpha=0.4)