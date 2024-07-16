"""
Okay we got some script from sina.. lets try it
"""

import sys
from objective_configuration.segment7T3T import DCODE_GAN
# Add the code path so we can import stuff frm there
sys.path.append(DCODE_GAN)

import torch
import helper.plot_class as hplotc
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

import re
from objective_configuration.segment7T3T import get_path_dict, DATASET_LIST
import sys
import numpy as np
import os
import nibabel
import helper.plot_class as hplotc
import glob
import helper.misc as hmisc
import argparse
import os
import helper.misc as hmisc

"""
Below are some options from Sina...

"""

# get test options
opt = TestOptions().parse()
print("PROCESSING VENDOR ", opt.vendor)
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

opt.model = 'cut'
# Change these..
# opt.image_dir_A = '/data/cmr7t3t/mms1/all_phases_mid/Vendor_A/Image_single_slice'
# opt.label_dir_A = '/data/cmr7t3t/mms1/all_phases_mid/Vendor_A/Label_single_slice'
# opt.image_dir_B = opt.image_dir_A
# opt.label_dir_B = opt.label_dir_A
opt.epoch = '185'
opt.dataset_mode = 'cmr'
opt.gpu_ids = [4]
opt.nce_idt = True
opt.amp = False
opt.direction = 'AtoB'
opt.lambda_identity = 0.5

opt.max_dataset_size = 5850
opt.output_nc = 1
opt.input_nc = 1
opt.batch_size = 1
opt.num_patches = 16

opt.display_freq = 50
opt.update_html_freq = 50
opt.save_epoch_freq = 10
opt.evaluation_freq = 500
opt.n_epochs = 50
opt.n_epochs_decay = 50
# If remote of course..
opt.checkpoints_dir = '/data/cmr7t3t/code/CMR_CUT_7T_Seb/checkpoints'
opt.log_file = opt.checkpoints_dir + '/' + opt.name +  '/loss_log.txt'
opt.loss_freq = 1000
opt.results_dir = '/data/cmr7t3t/mms1/test'

"""
Process my own input

"""


# argparse did not work nicely.. I added an option remotely
vendor = opt.vendor
if vendor == 'A':
    opt.name = 'new_seven_mms1A_cut_NCE8_GAN2_220907'
    path_dict = get_path_dict('mm1a')
    dsource = os.path.dirname(path_dict['dimg'])
    ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task903_GAN_MM1_A_balanced'
    # For the labels...
    linux_command_1 = f"cp {dsource}/labelsTr/* {ddest}/labelsTr"
    linux_command_2 = f"cp {dsource}/labelsTs/* {ddest}/labelsTs"
elif vendor == 'B':
    opt.name = 'new_seven_mms1B_cut_NCE8_GAN2_220907'
    path_dict = get_path_dict('mm1b')
    dsource = os.path.dirname(path_dict['dimg'])
    ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task904_GAN_MM1_B_balanced'
    # For the labels...
    linux_command_1 = f"cp {dsource}/labelsTr/* {ddest}/labelsTr"
    linux_command_2 = f"cp {dsource}/labelsTs/* {ddest}/labelsTs"
else:
    print('Unknown input. Exiting program')
    sys.exit()


hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=[])


"""
Now create stuff
"""
# Compatibility with Sina's code


def get_synthesize(model, data, index=0):
    if index == 0:
        model.data_dependent_initialize(data)
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        model.parallelize()
        if opt.eval:
            model.eval()

    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()  # get image results
    temp_gen = visuals['fake_B'].squeeze(dim=0).cpu().numpy()
    return temp_gen


# Train paths
source_path_img_train = os.path.join(dsource, 'imagesTr')
source_path_label_train = os.path.join(dsource, 'labelsTr')
target_path_img_train = os.path.join(ddest, 'imagesTr')
target_path_label_train = os.path.join(ddest, 'labelsTr')
# Test paths
source_path_img_test = os.path.join(dsource, 'imagesTs')
source_path_label_test = os.path.join(dsource, 'labelsTs')
target_path_img_test = os.path.join(ddest, 'imagesTs')
target_path_label_test = os.path.join(ddest, 'labelsTs')

# Just copying and fixing images now.. labels can be copied manually...
for datatype in ['train', 'test']:
    if datatype == 'train':
        source_path_img = os.path.join(dsource, 'imagesTr')
        source_path_label = os.path.join(dsource, 'labelsTr')
        target_path_img = os.path.join(ddest, 'imagesTr')
        target_path_label = os.path.join(ddest, 'labelsTr')
    else:
        source_path_img = os.path.join(dsource, 'imagesTs')
        source_path_label = os.path.join(dsource, 'labelsTs')
        target_path_img = os.path.join(ddest, 'imagesTs')
        target_path_label = os.path.join(ddest, 'labelsTs')

    image_file_list = os.listdir(source_path_img)
    opt.num_test = len(image_file_list)
    model = create_model(opt)      # create a model given opt.model and other options
    generated = []
    for i, i_file in enumerate(image_file_list):
        sel_img_file = os.path.join(source_path_img, i_file)
        single_image = hmisc.load_array(sel_img_file).T[:, ::-1, ::-1]
        single_tensor = torch.from_numpy(single_image[None].copy()).float()
        data = {'A': single_tensor, 'B': single_tensor, 'A_paths': '', 'B_paths': ''}
        file_synthesized = get_synthesize(model=model, data=data, index=i)
        # Now store stuff...
        nibabel_obj = nibabel.Nifti1Image(file_synthesized.T[::-1, ::-1], np.eye(4))
        nibabel.save(nibabel_obj, os.path.join(target_path_img, i_file))


# To copy the labels..
os.system(linux_command_1)
os.system(linux_command_2)