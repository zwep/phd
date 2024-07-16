"""
Also create a large set of GAN data
"""

# Count number of images we have

"""
Okay we got some script from sina.. lets try it
"""

import sys
import shutil
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
import helper.array_transf as harray
"""
Below are some options from Sina...

"""

# get test options
opt = TestOptions().parse()
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

opt.model = 'cut'
# Change these..
opt.image_dir_A = '/data/cmr7t3t/mms1/all_phases_mid/Vendor_A/Image_single_slice'
opt.label_dir_A = '/data/cmr7t3t/mms1/all_phases_mid/Vendor_A/Label_single_slice'
opt.image_dir_B = opt.image_dir_A
opt.label_dir_B = opt.label_dir_A
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
opt.log_file = opt.checkpoints_dir + '/' + opt.name + '/loss_log.txt'
opt.loss_freq = 1000
opt.results_dir = '/data/cmr7t3t/mms1/test'

"""
Create target dir
"""

# argparse did not work nicely.. Because of the code of Sina
ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task635_GAN_MM1_A_MM1_B_ACDC'
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=[])


"""
Now create stuff
"""


def get_synthesize(model, data, index=0):
    # With a dictionary as input, we get a synthesized verison back
    if index == 0:
        model.data_dependent_initialize(data)
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        model.parallelize()
        if opt.eval:
            model.eval()
#
    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()  # get image results
    temp_gen = visuals['fake_B'].squeeze(dim=0).cpu().numpy()
    return temp_gen


def store_n_items_with_gan(source_dir, target_dir, img_subdir, label_subdir, reduction_perc=None):
    if 'mm1_a' in source_dir.lower():
        max_epoch = 200
        prefix = 'gan_mm1a'
        opt.name = 'new_seven_mms1A_cut_NCE8_GAN2_220907'
    elif 'mm1_b' in source_dir.lower():
        max_epoch = 200
        opt.name = 'new_seven_mms1B_cut_NCE8_GAN2_220907'
        prefix = 'gan_mm1b'
    elif 'acdc' in source_dir.lower():
        max_epoch = 100
        prefix = 'gan_acdc'
        opt.name = 'seven2three_acdc_cut_NCE4_GAN2_np128_fe_211208'
    else:
        print('derp')
        sys.exit()
#
    img_dir = os.path.join(source_dir, img_subdir)
    label_dir = os.path.join(source_dir, label_subdir)
    dest_img_dir = os.path.join(target_dir, img_subdir)
    dest_label_dir = os.path.join(target_dir, label_subdir)
    file_list = os.listdir(img_dir)
    print('First create model statement')
#
    n_files = len(file_list)
    list_of_epoch = range(5, max_epoch, 5)
    if reduction_perc is not None:
        # Something like this....
        n_files = int(n_files * reduction_perc)

    print(f"Number of files {n_files}")
    print(f"Number of epochs {len(list_of_epoch)}")
    print(f"Expected number of images {n_files * len(list_of_epoch)}")
    for counter, sel_epoch in enumerate(list_of_epoch):
        counter_suffix = str(counter).zfill(2)
        epoch_str = str(sel_epoch).zfill(3)
        opt.epoch = str(sel_epoch)
        model = create_model(opt)
        dont_init_model = False
        for sel_index in range(n_files):
            sel_file_img = file_list[sel_index]
            file_ext = hmisc.get_ext(sel_file_img)
            sel_file_label = re.sub('_0000', '', sel_file_img)
            file_base_name = hmisc.get_base_name(sel_file_label)
    #
            # Define input and output file names
            source_img_file_path = os.path.join(img_dir, sel_file_img)
            source_label_file_path = os.path.join(label_dir, sel_file_label)
    #
            # Load array and convert to tensor..
            single_image = hmisc.load_array(source_img_file_path).T[:, ::-1, ::-1]
            single_label = hmisc.load_array(source_label_file_path).T[:, ::-1, ::-1]

            n_loc = single_image.shape[0]
            if n_loc > 3:
                range_locations = np.arange(n_loc // 2 - 2, n_loc // 2 + 2)
            else:
                range_locations = range(n_loc)
    #
            for ii, i_loc in enumerate(range_locations):
                img_file_name = f"{prefix}_{epoch_str}_{counter_suffix}_{file_base_name}_0000" + file_ext
                label_file_name = f"{prefix}_{epoch_str}_{counter_suffix}_{file_base_name}" + file_ext
                if n_loc > 1:
                    img_file_name = f"{prefix}_{epoch_str}_{counter_suffix}_{file_base_name}_loc_{i_loc}_0000" + file_ext
                    label_file_name = f"{prefix}_{epoch_str}_{counter_suffix}_{file_base_name}_loc_{i_loc}" + file_ext
    #
                target_img_file_path = os.path.join(dest_img_dir, img_file_name)
                target_label_file_path = os.path.join(dest_label_dir, label_file_name)
    #
                single_tensor = torch.from_numpy(single_image[i_loc:i_loc+1][None].copy()).float()
                data = {'A': single_tensor, 'B': single_tensor, 'A_paths': '', 'B_paths': ''}
                file_synthesized = get_synthesize(model=model, data=data, index=dont_init_model)
                # file_synthesized = file_synthesized
                # Store img
                nibabel_obj = nibabel.Nifti1Image(file_synthesized.T[::-1, ::-1], np.eye(4))
                nibabel.save(nibabel_obj, target_img_file_path)
                # Store label
                nibabel_obj = nibabel.Nifti1Image(single_label[i_loc:i_loc+1].T[::-1, ::-1], np.eye(4))
                nibabel.save(nibabel_obj, target_label_file_path)

                dont_init_model = True
    #
                counter += 1


# Train paths
source_path_mm1a = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task501_MM1_A'
source_path_mm1b = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task502_MM1_B'
source_path_acdc = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task511_ACDC'
target_path = ddest

#
# # Store train/test
# n_epoch = None
store_n_items_with_gan(source_path_mm1a, ddest, 'imagesTr', 'labelsTr')
store_n_items_with_gan(source_path_mm1a, ddest, 'imagesTs', 'labelsTs', reduction_perc=0.3)

store_n_items_with_gan(source_path_mm1b, ddest, 'imagesTr', 'labelsTr')
store_n_items_with_gan(source_path_mm1b, ddest, 'imagesTs', 'labelsTs', reduction_perc=0.3)

store_n_items_with_gan(source_path_acdc, ddest, 'imagesTr', 'labelsTr')
store_n_items_with_gan(source_path_acdc, ddest, 'imagesTs', 'labelsTs', reduction_perc=0.3)

