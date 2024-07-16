
import re
import shutil
import os
import argparse
from objective_configuration.segment7T3T import TASK_NR_TO_DIR, get_path_dict, DATASET_LIST


"""
Now that we have a single example.. lets evaluate..

"""

parser = argparse.ArgumentParser()
parser.add_argument('-t', type=str)
parser.add_argument('-f', type=str, default='-1')
parser.add_argument('-dataset', type=str, default=None)


# Parses the input
p_args = parser.parse_args()
task_number = p_args.t
fold_parameter = p_args.f
sel_dataset = p_args.dataset

task_dir = TASK_NR_TO_DIR.get(task_number.zfill(3), 'Unknown')

dsource = '/data/seb/data/mm_christina/preproc_image'
dtarget = '/data/seb/data/mm_christina/segm_result'

cmd_line = f"nnUNet_predict -i {dsource}  -o {dtarget} -t {task_number} -m 2d -f {fold_parameter} --overwrite_existing"
os.system(cmd_line)