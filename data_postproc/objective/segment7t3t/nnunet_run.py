"""
This runs the prediction of a nnUnet task -t and stores the predictions in the right location
"""

import re
import shutil
import os
import argparse
from objective_configuration.segment7T3T import TASK_NR_TO_DIR, get_path_dict, DATASET_LIST, DCODE

# Execute prediction...
# -i shows the input...
# -o the output dir..
# -t the Task number

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
if sel_dataset is not None:
    DATASET_LIST = [sel_dataset]

for dataset in DATASET_LIST:
    path_dict = get_path_dict(dataset)
    input_dir = path_dict['dimg']
    output_dir = os.path.join(path_dict['dresults'], task_dir)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    else:
        os.makedirs(output_dir)

    print(f"Predicting {dataset} images")
    print(f"\t Task number: {task_number}")
    print(f"\t Fold option: {fold_parameter}")
    print(f"\t Input directory: {input_dir}")
    print(f"\t Output directory: {output_dir}")

    if fold_parameter == '-1':
        cmd_line = f"nnUNet_predict -i {input_dir}  -o {output_dir} -t {task_number} -m 2d --overwrite_existing"
    else:
        cmd_line = f"nnUNet_predict -i {input_dir}  -o {output_dir} -t {task_number} -m 2d -f {fold_parameter} --overwrite_existing"

    os.system(cmd_line)

    print("=== done inference using nnUnet ===")
    print("=== Start inference using nnUnet ===")

    # We now give the exact task number so that we are sure that we evaluate the correct model output.
    # This prevents any miscommunication when doing stuff in parallel
    cmd_line = f'python {DCODE}/data_postproc/objective/segment7t3t/measure_model_metric.py -dataset {dataset} -model t{task_number}'
    os.system(cmd_line)