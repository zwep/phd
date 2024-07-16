"""
The commands below can be run remotely where we have the nnunet package installed in the venv
Here it doesnt make much sense to run it.
"""
import numpy as np
from nnunet.dataset_conversion.utils import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import os
import argparse
import re
from objective_configuration.segment7T3T import TASK_NR_TO_DIR

parser = argparse.ArgumentParser()
parser.add_argument('-t', type=str)

# Parses the input
p_args = parser.parse_args()
task_number = p_args.t
task_number = task_number.zfill(3)
task_dir = TASK_NR_TO_DIR.get(task_number, None)

if task_dir is None:
    print("Unknown task. Received: ", task_number)
    print("Need any of : ", sorted(list(TASK_NR_TO_DIR.keys())))
else:
    print("Received task number  ", task_number)
    print("Associated task dir ", task_dir)
    target_base = os.path.join(nnUNet_raw_data, task_dir)
    target_imagesTr = os.path.join(target_base, "imagesTr")
    target_imagesTs = os.path.join(target_base, "imagesTs")
    target_labelsTs = os.path.join(target_base, "labelsTs")
    target_labelsTr = os.path.join(target_base, "labelsTr")
    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(os.path.join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Rho',),
                          labels={0: 'background', 1: 'RV', 2: 'MYO', 3: 'LV'}, dataset_name=task_dir, license='It is I, Seb')
