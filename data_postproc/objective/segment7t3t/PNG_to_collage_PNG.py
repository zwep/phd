"""
I want to make a nice collage of certain image paths...
"""

import sys
sys.path.append('/home/bugger/PycharmProjects/pytorch_in_mri')
import argparse
import os
import helper.misc as hmisc
import numpy as np
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

model_name_list = [x for x in os.listdir(ddata_PNG) if os.path.isdir(os.path.join(ddata_PNG, x))]
model_name_list = sorted(model_name_list, key=lambda x: os.path.getmtime(os.path.join(ddata_PNG, x)))[::-1]
model_name_list = np.array(model_name_list)

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
    file_list_png = [os.path.join(model_dir_path, x) for x in file_list_png if not x.startswith('collage')]
    ddest_model = os.path.join(ddata_PNG, i_model_name_dir)
    plot_obj = hplotc.PlotCollage(content_list=file_list_png, ddest=ddest_model, n_display=6)
    plot_obj.plot_collage()