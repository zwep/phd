import re
import sys
import numpy as np
import os
import nibabel
import data_generator.Segment7T3T as dg_segment_7t3t
import helper.plot_class as hplotc
import glob
import helper.misc as hmisc
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-vendor', type=str)
parser.add_argument('-data', type=str)
p_args = parser.parse_args()
vendor = p_args.vendor
datatype = p_args.data

"""

"""

ddata = f'/data/cmr7t3t/biasfield_sa_mm1_{vendor}'
if vendor == 'A':
    ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task601_Biasfield_MM1_A'
elif vendor == 'B':
    ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task603_Biasfield_MM1_B'
else:
    print('Unknown input. Exiting program')
    sys.exit()

hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=[])

if datatype in ['test', 'train']:
    dataset_type = datatype  # Change this to train / test
else:
    print('Unknown data type. Exit program: ', datatype)
    sys.exit()

dg_obj = dg_segment_7t3t.DataGeneratorCardiacSegment(ddata=ddata,
                                                     dataset_type=dataset_type,
                                                     target_type='segmentation',
                                                     transform_resize=True,
                                                     transform_type="abs")
dg_obj.resize_list = [(256, 256)]
dg_obj.resize_index = 0
file_list = dg_obj.container_file_info[0]['file_list']
if dataset_type == 'train':
    ddest_img = os.path.join(ddest, 'imagesTr')
    ddest_label = os.path.join(ddest, 'labelsTr')
else:
    ddest_img = os.path.join(ddest, 'imagesTs')
    ddest_label = os.path.join(ddest, 'labelsTs')


if os.path.isdir(ddest_img):
    files = glob.glob(ddest_img + '/*')
    [os.remove(f) for f in files]
    files = glob.glob(ddest_label + '/*')
    [os.remove(f) for f in files]
else:
    os.makedirs(ddest_img)
    os.makedirs(ddest_label)


for sel_item in range(len(dg_obj)):
    cont = dg_obj.__getitem__(sel_item)
    file_name = cont['file_name']  # This still has the extension .npy
    base_name = hmisc.get_base_name(file_name)
    #
    input = np.array(cont['input'])
    temp_target = np.array(cont['target'])
    # Convert segmentation from binary to integer
    x_padded = np.concatenate([np.zeros(temp_target.shape[-2:])[None], temp_target])
    x_rounded = np.isclose(x_padded, 1, atol=0.8).astype(int)
    target = np.argmax(x_rounded, axis=0)[np.newaxis]
    # Swap the labels. This is needed vor MM1
    # This is not necessary anymore....
    # label_index_1 = target == 1
    # label_index_3 = target == 3
    # target[label_index_1] = 3
    # target[label_index_3] = 1
    # print("Input shape ", input.shape)g
    # print("Target shape ", target.shape)
    # fig_obj = hplotc.ListPlot([input, target])
    # fig_obj.figure.savefig(f'/data/seb/inp_target_{sel_item}.png')
    # Convert to nibabel
    input_nibabel_obj = nibabel.Nifti1Image(input.T[::-1, ::-1], np.eye(4))
    target_nibabel_obj = nibabel.Nifti1Image(target.T[::-1, ::-1], np.eye(4))
    # Store the nibabel objects
    # print(f"Storing in {ddest_img}")
    nibabel.save(input_nibabel_obj, os.path.join(ddest_img, f'{base_name}_0000.nii.gz'))
    nibabel.save(target_nibabel_obj, os.path.join(ddest_label, f'{base_name}.nii.gz'))
