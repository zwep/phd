import sys
sys.path.append('/home/bugger/PycharmProjects/pytorch_in_mri')
import argparse
import os
import numpy as np
from objective_configuration.segment7T3T import get_path_dict, TRANSFORM_MODEL_NAMES, DIR_TO_TASK_NR
import helper.plot_class as hplotc

"""
Different tactic... here we are going to compare the outcome of different models

Also always add the ground truth to it..?
"""

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str)
parser.add_argument('-model', type=str, default=None)
parser.add_argument('-n', type=str, default=None)
p_args = parser.parse_args()
dataset = p_args.dataset
model_selection = p_args.model
n_examples = p_args.n

path_dict = get_path_dict(dataset)

ddata_PNG = path_dict['dpng']

model_name_list = [x for x in os.listdir(ddata_PNG) if os.path.isdir(os.path.join(ddata_PNG, x))]
model_name_list = [x for x in model_name_list if not x.startswith('collage')]
model_name_list = sorted(model_name_list, key=lambda x: os.path.getmtime(os.path.join(ddata_PNG, x)))[::-1]
model_name_list = np.array(model_name_list)


import objective_helper.segment7T3T as hsegm7t
if model_selection:
    sel_model_name_list = hsegm7t.model_selection_processor(model_selection, model_name_list)
else:
    print("Please select a model first..")
    sys.exit()

sel_model_name_list = ['ground_truth'] + sel_model_name_list
# For files.. loop over models..
model_dir_path = os.path.join(ddata_PNG, sel_model_name_list[0])
sub_ddest = 'collage_' + sel_model_name_list[1] + '_' + sel_model_name_list[-1]
ddest = os.path.join(ddata_PNG, sub_ddest)
if not os.path.isdir(ddest):
    os.makedirs(ddest)

# Get all the sets and their png stuff
file_set_list = []
for i_model_name in sel_model_name_list:
    temp_path = os.path.join(ddata_PNG, i_model_name)
    temp_file_list = os.listdir(temp_path)
    temp_file_list = [x for x in temp_file_list if not x.startswith('collage')]
    file_set_list.append(set(temp_file_list))

file_list_png = list(set.intersection(*file_set_list))
if n_examples is None:
    n_examples = len(file_list_png)
else:
    n_examples = int(n_examples)

# Build full model thing
plot_file_list = []
for i_file in file_list_png:
    for i_model in sel_model_name_list:
        temp_file = os.path.join(ddata_PNG, i_model, i_file)
        plot_file_list.append(temp_file)

n_models = len(sel_model_name_list)
plot_file_list = plot_file_list[:int(n_examples * n_models)]
neat_model_names = [TRANSFORM_MODEL_NAMES.get(DIR_TO_TASK_NR.get(x, 'No Name'), 'No Name')for x in sel_model_name_list]
plot_obj = hplotc.PlotCollage(content_list=plot_file_list, ddest=ddest, n_display=n_models,
                              subtitle_list=neat_model_names)
plot_obj.plot_collage()
