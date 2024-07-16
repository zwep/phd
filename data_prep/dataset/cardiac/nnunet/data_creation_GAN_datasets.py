
import os
import sys
import helper.misc as hmisc
from objective_configuration.segment7T3T import DCMR7T
from nnunet.paths import nnUNet_raw_data
import numpy as np
import nibabel

"""
We have some GAN data...
We'd like to use that during training.. We have setup several experiments:

Start experimenten met 7T GAN data...
601 - GAN Copy from ... /data/cmr7t3t/results/ACDC_220121
"""


def load_and_correct_dimensions(file_name):
    affine_struct = nibabel.load(file_name).affine
    file_array = hmisc.load_array(file_name)
    file_array = np.squeeze(file_array)
    # Previously we used the affine struct.... but that caused weird problems.. Now we do this
    nibabel_obj = nibabel.Nifti1Image(file_array, np.eye(4))
    nibabel.save(nibabel_obj, file_name)


# Here lies the ACDC-GAN data..
temp_dimg = 'results/ACDC_220121/seven2three_acdc_cut_NCE4_GAN2_np128_fe_211208/test_100/niftis/cmr3t2cmr7t'
dimg = os.path.join(DCMR7T, temp_dimg)
temp_dlabel = 'results/ACDC_220121/seven2three_acdc_cut_NCE4_GAN2_np128_fe_211208/test_100/niftis/cmr3t_label'
dlabel = os.path.join(DCMR7T, temp_dlabel)


ddest = 'Task610_GAN_ACDC'
dsubdir = ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']
raw_dest = os.path.join(nnUNet_raw_data, ddest)
hmisc.create_datagen_dir(raw_dest, type_list=dsubdir, data_list=[])

# Copy the GAN files to my /data/nnunet folder
dimg_train = os.path.join(nnUNet_raw_data, ddest, 'imagesTr')
dlabel_train = os.path.join(nnUNet_raw_data, ddest, 'labelsTr')
dimg_test = os.path.join(nnUNet_raw_data, ddest, 'imagesTs')
dlabel_test = os.path.join(nnUNet_raw_data, ddest, 'labelsTs')

os.system(f"cp -r {dimg}/* {dimg_train}")
os.system(f"cp -r {dlabel}/* {dlabel_train}")

# We want to correct the dimensions too, now they are 4D..
for i_file in os.listdir(dimg_train):
    full_path = os.path.join(dimg_train, i_file)
    load_and_correct_dimensions(full_path)
    full_path = os.path.join(dlabel_train, i_file)
    load_and_correct_dimensions(full_path)

# I need to rename the images... append _0000
# And copy patient 060-070 to test
for i_file in os.listdir(dimg_train):
    base_name = hmisc.get_base_name(i_file)
    ext = hmisc.get_ext(i_file)
    dsource = os.path.join(dimg_train, i_file)
    if i_file.startswith('patient06') or i_file.startswith('patient07'):
        dtarget = os.path.join(dimg_test, base_name + "_0000" + ext)
    else:
        dtarget = os.path.join(dimg_train, base_name + "_0000" + ext)

    os.rename(dsource, dtarget)

# And copy patient 060-070 to test
for i_file in os.listdir(dlabel_train):
    dsource = os.path.join(dlabel_train, i_file)
    if i_file.startswith('patient06') or i_file.startswith('patient07'):
        dtarget = os.path.join(dlabel_test, i_file)
        os.rename(dsource, dtarget)


username = os.environ.get('USER', os.environ.get('USERNAME'))
# Only do this when we are local...
# This was a quick fix to change the orientation of the data...
import re
dstart = '/home/bugger/Documents/data/segmen7T3T/Task610_GAN_ACDC'
if username == 'bugger':
    for dsource, _, f in os.walk(dstart):
        filter_f = [x for x in f if x.endswith('nii.gz')]
        if len(filter_f):
            for i_file in filter_f:
                dsource_file = os.path.join(dsource, i_file)
                dtarget = re.sub('Task610_GAN_ACDC',
                                 'Task610_GAN_ACDC_again', dsource)
                dtarget_file = os.path.join(dtarget, i_file)
                if not os.path.isdir(dtarget):
                    os.makedirs(dtarget)
                nib_obj = nibabel.load(dsource_file)
                struct_matrix = np.eye(4)
                # struct_matrix[0][0] = 2
                # struct_matrix[1][1] = 2
                nibabel_obj = nibabel.Nifti1Image(nib_obj.get_fdata(), struct_matrix)
                print(dsource_file)
                print(dtarget_file)
                print(struct_matrix, nib_obj.affine)
                nibabel.save(nibabel_obj, dtarget_file)