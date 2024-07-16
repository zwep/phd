import re
import os
import argparse
from nnunet.paths import nnUNet_raw_data, nnUNet_cropped_data, \
    preprocessing_output_dir, network_training_output_dir_base
import shutil

from objective_configuration.segment7T3T import TASK_NR_TO_DIR
from objective_configuration.segment7T3T import get_path_dict, DATASET_LIST

"""
The removes the old folders of a task

Created by Seb Harrevelt - 2022 nov
"""

parser = argparse.ArgumentParser()
parser.add_argument('-t', type=str)
parser.add_argument('-all', type=bool)

p_args = parser.parse_args()
task_number = p_args.t
yes_to_all = p_args.all
task_dir = TASK_NR_TO_DIR.get(task_number.zfill(3), 'Unknown')


result_paths = [get_path_dict(idataset)['dresults'] for idataset in DATASET_LIST]

for i_dir in [nnUNet_cropped_data, preprocessing_output_dir, network_training_output_dir_base] + result_paths:
    if 'trained_models' in i_dir:
        full_task_dir = os.path.join(i_dir, 'nnUNet/2d', task_dir)
    else:
        full_task_dir = os.path.join(i_dir, task_dir)

    print("Trying to delete ", full_task_dir)
    if os.path.isdir(full_task_dir):
        if yes_to_all:
            answer = 'y'
        else:
            print(f"Do you want to delete {full_task_dir}?")
            answer = input('Answer y/n')
        if answer.lower() == 'y':
            print('Deleting the directory')
            shutil.rmtree(full_task_dir)
        else:
            print('Skip the delete')
    else:
        print('No directory found: ', full_task_dir)