import shutil
import os
from objective_configuration.segment7T3T import get_path_dict, DATASET_LIST, DFINAL, DLOG
import argparse
from loguru import logger

logger.add(os.path.join(DLOG, "copy_resulting_figures.log"))

parser = argparse.ArgumentParser()
parser.add_argument('-name', '-n', type=str)
parser.add_argument('-dataset', type=str)

# Parses the input
p_args = parser.parse_args()
dest_name = p_args.name
dataset = p_args.dataset

"""
With another script we have created a lot of figures. These are the resulting figures of a metric calculatoin.
If we run that same script with different settings, it will override those figures.

Therefore we want to move them to a different place
"""
all_dataset = ['mm1a', 'mm1b', 'acdc', '7t', 'kaggle', 'mm2']
subset_dataset = ['7t', 'kaggle', 'mm2']

sel_dataset = subset_dataset
if dataset == 'all':
    sel_dataset = all_dataset

for i_dataset in sel_dataset:
    logger.debug(f"\n\nStarting with data source {i_dataset}")
    path_dict = get_path_dict(i_dataset)
    # Source paths
    dresults = path_dict['dresults']
    # Target path
    dtarget = os.path.join(DFINAL, dest_name)
    if not os.path.isdir(dtarget):
        logger.debug(f"The target directory is created: {dtarget}")
        os.makedirs(dtarget)
    else:
        logger.debug(f"The target directory exists: {dtarget}")

    # Copy these file names
    files_to_copy = ['hausdorf_per_model_per_class_boxplot.png', 'hausdorf_per_model_subject_class.csv',
                     'dice score_per_model_per_class_boxplot.png', 'dice score_per_model_subject_class.csv']

    for i_file in files_to_copy:
        logger.debug(f"Copy file name {i_file}")
        src_file = os.path.join(dresults, i_file)
        tgt_file = os.path.join(dtarget, f'{i_dataset}:' + i_file)
        logger.debug(f"Source file {src_file}")
        logger.debug(f"Target file name {tgt_file}")
        shutil.copy(src_file, tgt_file)
