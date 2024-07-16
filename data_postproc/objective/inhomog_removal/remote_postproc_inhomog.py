"""
Remote evaluation... for the test set..

also the data paths should be set to the test split
"""

import objective.inhomog_removal.executor_inhomog_removal as executor
import objective.inhomog_removal.postproc_inhomog_removal as postproc_inhomog
import helper.plot_class as hplotc
import os
from loguru import logger
# needed these for some misc coding
import helper.array_transf as harray
import matplotlib.pyplot as plt
import numpy as np
import sys
import helper.misc as hmisc
import argparse
from objective_configuration.inhomog_removal import MODEL_DIR, get_path_dict, LOG_DIR, IMG_SYNTH_COIL


try:
    file_base_name = hmisc.get_base_name(__file__)
    logger.add(os.path.join(LOG_DIR, f"{file_base_name}.log"))
except NameError:
    print('No file name known. Not reporting to logger.')


parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, help='Provide the name of the directory that we want to post process')
parser.add_argument('-dataset', type=str, default='all',
                    help='Provide the name of the dataset on which we want to evaluate: '
                         'synthetic, 3T, patient, volunteer')
parser.add_argument('-debug', type=bool, default=False)


p_args = parser.parse_args()
path = p_args.path
dataset = p_args.dataset
debug = p_args.debug

# Get all paths..
dconfig = os.path.join(MODEL_DIR, path)
path_dict, dataset_list = get_path_dict(dconfig)

# Check what kind of synthetic data input we need (multi coil or single)
single_input = True
if 'multi' in path:
    single_input = False
    # Use the individual coil images path as input
    # Why..?
    # path_dict['synthetic']['dimage'] = IMG_SYNTH_COIL
    # Limit the number of datasets we check
    dataset_list = ['volunteer', 'synthetic']

if dataset == 'all':
    sel_dataset_list = dataset_list
else:
    if dataset in dataset_list:
        sel_dataset_list = [dataset]
    else:
        logger.debug(f'Unknown dataset selected: {dataset}')
        sys.exit()


for idataset in sel_dataset_list:
    temp_dict = path_dict[idataset]
    base_name = os.path.basename(temp_dict['dpred'])
    logger.debug(f'Processing {base_name}')
    hmisc.create_datagen_dir(temp_dict['dpred'], data_list=[], type_list=['input', 'biasfield', 'pred', 'mask', 'target'])
    mask_ext = '.nii.gz'
    mask_suffix = ''
    stride = 64
    if 'volunteer' in base_name:
        mask_ext = '.npy'
        mask_suffix = ''
        # This used to be nii.gz... I changed it to npy
        file_list = ["v9_03032021_1647583_11_3_t2wV4.npy",
                     "v9_11022021_1643158_8_3_t2wV4.npy",
                     "v9_10022021_1725072_12_3_t2wV4.npy",
                     "v9_18012021_0939588_10_3_t2wV4.npy"]
    elif base_name.endswith('3T'):
        mask_ext = '.h5'
        mask_suffix = '_target'
        # I dont know why I gave these the nii.gz extension
        # file_list = ['8_MR.nii.gz',
        #              '19_MR.nii.gz',
        #              '41_MR.nii.gz',
        #              '45_MR.nii.gz']
        file_list = ['8_MR.h5',
                     '19_MR.h5',
                     '41_MR.h5',
                     '45_MR.h5']
    elif base_name.endswith('patient_corrected'):
        mask_ext = '.npy'
        stride = 128
        file_list = ["7TMRI002.npy", "7TMRI005.npy", "7TMRI016.npy", "7TMRI020.npy"]
    elif base_name == 'target_corrected':
        mask_ext = '.nii.gz'
        file_list = ['M20_to_4_MR_20210107_0002_transversal', 'M21_to_5_MR_20210329_0002_transversal',
                     'M22_to_6_MR_20210312_0002_transversal', 'M23_to_7_MR_20210108_0002_transversal']
        if single_input:
            file_list = [x + '.nii.gz' for x in file_list]
        else:
            file_list = [x + '.npy' for x in file_list]
    else:
        logger.debug('File list is empty')
        file_list = []

    target_dir = temp_dict.get('dtarget', None)
    postproc_obj = postproc_inhomog.PostProcInhomogRemoval(image_dir=temp_dict['dimage'],
                                                           mask_dir=temp_dict['dmask'],
                                                           dest_dir=temp_dict['dpred'],
                                                           target_dir=target_dir,
                                                           config_path=dconfig,
                                                           executor_module=executor, config_name='config_param.json',
                                                           stride=stride, patch_shape=(256, 256),
                                                           storage_extension='nii',
                                                           mask_ext=mask_ext,
                                                           mask_suffix=mask_suffix)
    # This experimental post proc is related to the models that produce both the homogeneous and bias field image
    # Number 3 really produces nice images...
    # But is in theory IMO not the best... we are averaging anatomical images..
    # Number 2 averages the bias fields.. which should, in my idea, work better..
    # postproc_obj.experimental_postproc_both = 3
    # Run only specific files..
    if debug:
        logger.debug(f'Running on the following files')
        for file_name in file_list:
            logger.debug(f'\t {file_name}')

        postproc_obj.file_list = file_list
    postproc_obj.run()
    # import time
    # postproc_obj.load_file(0)
    # t0 = time.time()
    # postproc_obj.run_slice_patched(0)
    # print(time.time() - t0)
    # # postproc_obj.run_iterative_recon()

