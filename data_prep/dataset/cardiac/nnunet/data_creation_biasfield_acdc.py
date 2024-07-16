import re
import numpy as np
import os
import nibabel
import data_generator.Segment7T3T as dg_segment_7t3t
import helper.plot_class as hplotc

"""
Here we create the data using the DataGenerator from Segment7T3T biasfield
-> We store this in the nnUNet_raw_data folder

-- This is also needed somewhere...
data_prep/dataset/cardiac/nnunet/dataset_json_creation.py -t XXX

After this we need to verify the data create
    nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity
This will populate the following folder: nnUNet_preprocessed

Training can be commenced by calling
    nnUNet_train 2d nnUNetTrainerV2 TaskXXX_MYTASK FOLD --npz

export CUDA_VISIBLE_DEVICES=6
nnUNet_train 2d nnUNetTrainerV2 466 1 --npz

nnUNet_train 2d nnUNetTrainerV2 Task511_ACDC 0 --npz
nnUNet_train 2d nnUNetTrainerV2 Task611_Biasfield_ACDC 0 --npz

This is the possible command for using pretrained weight and continue training.
nnUNet_train 2d nnUNetTrainerV2 Task611_Biasfield_ACDC 0 --npz -pretrained_weights  /data/seb/nnunet/nnUNet_trained_models/nnUNet/2d/Task999_7T/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model
"""

ddata = '/data/cmr7t3t/biasfield_sa_acdc'
ddest = '/data/s/nnUNet_raw_data/Task611_Biasfield_ACDC'

dataset_type = 'train'  # Change this to train / test
dg_obj = dg_segment_7t3t.DataGeneratorCardiacSegment(ddata=ddata,
                                                     dataset_type=dataset_type, target_type='segmentation',
                                                     transform_resize=True,
                                                     transform_type="abs")
dg_obj.resize_list = [(256, 256)]
dg_obj.resize_index = 0
file_list = dg_obj.container_file_info[0]['file_list']
patient_id_number = [int(re.findall("patient([0-9]*)", x)[0]) for x in file_list]
patient_id_number = np.array(patient_id_number)
# Filter out patients with id..
if dataset_type == 'train':
    filter_id = np.argwhere(patient_id_number <= 69)
    ddest_img = os.path.join(ddest, 'imagesTr')
    ddest_label = os.path.join(ddest, 'labelsTr')
    ddata_acdc_img = '/data/cmr7t3t/acdc/acdc_processed/Image'
    ddata_acdc_label = '/data/cmr7t3t/acdc/acdc_processed/Label'
else:
    filter_id = np.argwhere(patient_id_number > 69)
    ddest_img = os.path.join(ddest, 'imagesTs')
    ddest_label = os.path.join(ddest, 'labelsTs')
    ddata_acdc_img = '/data/cmr7t3t/acdc/acdc_processed/ImageTest'
    ddata_acdc_label = '/data/cmr7t3t/acdc/acdc_processed/LabelTest'

filtered_file_list = np.array(file_list)[filter_id]
filtered_file_list = list(filtered_file_list.ravel())

dg_obj.container_file_info[0]['file_list'] = filtered_file_list

for sel_item in range(len(dg_obj)):
    # for sel_item in range(len(dg_obj)):
    cont = dg_obj.__getitem__(sel_item)
    input = np.array(cont['input'])
    # Convert segmentation from binary to integer
    temp_target = np.array(cont['target'])
    x_padded = np.concatenate([np.zeros(temp_target.shape[-2:])[None], temp_target])
    x_rounded = np.isclose(x_padded, 1, atol=0.8).astype(int)
    target = np.argmax(x_rounded, axis=0)[np.newaxis]
    file_name = cont['file_name']  # This still has the extension .npy
    # Get frame and patient id..
    patient_id = int(re.findall("patient([0-9]*)", file_name)[0])
    frame_id = int(re.findall("frame([0-9]*)", file_name)[0])
    file_name_cropped = re.findall("(patient.*)\.", file_name)[0]
    acdc_file_name = f'patient{str(patient_id).zfill(3)}_frame{str(frame_id).zfill(2)}.nii.gz'
    acdc_file_path = os.path.join(ddata_acdc_img, acdc_file_name)
    # This is needed to get the affine matrix
    acdc_img = nibabel.load(acdc_file_path)
    # I assume that both input and target share the same affine matrix
    # The transpose is needed here, see: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md
    input_nibabel_obj = nibabel.Nifti1Image(input.T, acdc_img.affine)
    target_nibabel_obj = nibabel.Nifti1Image(target.T, acdc_img.affine)
    # Now do the file stuff
    nibabel.save(input_nibabel_obj, os.path.join(ddest_img, f'{file_name_cropped}_0000.nii.gz'))
    nibabel.save(target_nibabel_obj, os.path.join(ddest_label, f'{file_name_cropped}.nii.gz'))
