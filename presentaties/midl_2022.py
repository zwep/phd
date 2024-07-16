"""
I want to make a nice collage of certain image paths...
"""

import sys
sys.path.append('/home/bugger/PycharmProjects/pytorch_in_mri')
import argparse
import json
import helper.array_transf as harray
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm
import pathlib
import helper.misc as hmisc
import matplotlib.font_manager
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import ImageColor, Image
import helper.plot_class as hplotc
from objective_configuration.segment7T3T import get_path_dict

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str)
parser.add_argument('-model', type=str, default=None)
p_args = parser.parse_args()
dataset = p_args.dataset
model_selection = p_args.model

path_dict = get_path_dict(dataset)

ddata_PNG = path_dict['dpng']

model_name_list = np.array([x for x in os.listdir(ddata_PNG) if os.path.isdir(os.path.join(ddata_PNG, x))])
print("List of model names:")
for i, imodelname in enumerate(model_name_list):
    print(i, '\t', imodelname)

import objective_helper.segment7T3T as hsegm7t
if model_selection:
    sel_model_name_list = hsegm7t.model_selection_processor(model_selection, model_name_list)
else:
    print("Please select a model first..")
    sys.exit()



for i_model_name_dir in sel_model_name_list:
    print("Segmentation directory ", i_model_name_dir)
    model_dir_path = os.path.join(ddata_PNG, i_model_name_dir)
    file_list_png = os.listdir(model_dir_path)
    file_list_png = [x for x in file_list_png if not x.startswith('collage')]
    n_files = len(file_list_png)
    multp_six = n_files // 6
    n_files_multp_six = multp_six * 6
    counter = -1
    for index_range in np.split(np.arange(n_files_multp_six), n_files_multp_six//6):
        plot_array = []
        file_string = ''
        for ii in index_range:
            sel_file = file_list_png[ii]
            base_name = hmisc.get_base_name(sel_file)
            file_png = os.path.join(model_dir_path, sel_file)
            png_array = hmisc.load_array(file_png)
            plot_array.append(png_array)
            file_string += base_name + "_"
        dest_file_name = os.path.join(ddata_PNG, i_model_name_dir, 'collage_' + file_string[:-1] + '.png')
        plot_array = np.stack(plot_array)
        fig_obj = hplotc.ListPlot([plot_array], cmap='rgb', debug=True,
                                  sub_col_row=(2, 3), ax_off=True, wspace=0, hspace=0,
                                  figsize=(10, 15), aspect='auto')
        fig_obj.figure.savefig(dest_file_name, bbox_inches='tight', pad_inches=0.0)
