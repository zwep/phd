import numpy as np
import helper.plot_class as hplotc
import os
import nibabel
import helper.misc as hmisc
import argparse

"""
We want ground truth labels for 7T data

NNunet gives proper results

Used commands to generate predictions

# This is the 2d approach. Works best
# Here in the imageTs folder we have copied the 7T images...
nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task999_7T/imagesTs/ -o /data/seb/data/result_acdc -t 27 --save_npz -m 2d

# The following can be used to predict segmentations on the trained nnunet model with biasfield data
# Same test location (this contains the 7T images) but different Task name..
nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task999_7T/imagesTs/ -o /data/seb/data/result_acdc -t 599 --save_npz -m 2d

nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task999_7T/imagesTs/ -o /data/seb/data/result_acdc -t 599 --save_npz -m 2d

# Test image acdc...
nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task999_7T/imagesTs_acdc/ -o /data/cmr7t3t/acdc/acdc_processed/Results/nn_unet_base_line -t 27 --save_npz -m 2d

# Here we got the 3d one
nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task999_7T/imagesTs/ -o /data/seb/data/result_acdc_3d -t 27 --save_npz -m 3d_fullres

With this we can create simple images out of them to check the performance

# Maybe I can use this for the prediction of the trained task
nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task999_7T/imagesTs/ -o /data/seb/data/result_biasfield_acdc -t 599 --save_npz -m 2d
"""

# ??
# ['patient082_frame07.nii.gz', 'patient083_frame08.nii.gz', 'patient074_frame01.nii.gz', 'patient087_frame10.nii.gz', 'patient086_frame01.nii.gz', 'patient077_frame01.nii.gz', 'patient093_frame01.nii.gz', 'patient092_frame06.nii.gz', 'patient077_frame09.nii.gz', 'patient098_frame01.nii.gz', 'patient099_frame01.nii.gz', 'patient075_frame06.nii.gz', 'patient095_frame01.nii.gz', 'patient076_frame01.nii.gz', 'patient100_frame01.nii.gz', 'patient099_frame09.nii.gz', 'patient089_frame10.nii.gz', 'patient073_frame01.nii.gz', 'patient080_frame10.nii.gz', 'patient088_frame12.nii.gz', 'patient083_frame01.nii.gz', 'patient080_frame01.nii.gz', 'patient095_frame12.nii.gz', 'patient092_frame01.nii.gz', 'patient100_frame13.nii.gz', 'patient096_frame08.nii.gz', 'patient084_frame01.nii.gz', 'patient070_frame01.nii.gz', 'patient093_frame14.nii.gz', 'patient084_frame10.nii.gz', 'patient072_frame11.nii.gz', 'patient087_frame01.nii.gz', 'patient091_frame09.nii.gz', 'patient085_frame01.nii.gz', 'patient094_frame01.nii.gz', 'patient094_frame07.nii.gz', 'patient097_frame01.nii.gz', 'patient091_frame01.nii.gz', 'patient082_frame01.nii.gz', 'patient085_frame09.nii.gz', 'patient079_frame11.nii.gz', 'patient075_frame01.nii.gz', 'patient097_frame11.nii.gz', 'patient089_frame01.nii.gz', 'patient076_frame12.nii.gz', 'patient078_frame09.nii.gz', 'patient096_frame01.nii.gz', 'patient074_frame12.nii.gz', 'patient072_frame01.nii.gz', 'patient090_frame04.nii.gz', 'patient088_frame01.nii.gz', 'patient078_frame01.nii.gz', 'patient071_frame01.nii.gz', 'patient070_frame10.nii.gz', 'patient081_frame01.nii.gz', 'patient098_frame09.nii.gz', 'patient073_frame10.nii.gz', 'patient071_frame09.nii.gz', 'patient081_frame07.nii.gz', 'patient086_frame08.nii.gz', 'patient090_frame11.nii.gz', 'patient079_frame01.nii.gz']

# Visualize initial predictions from nn unet

parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str)

# Parses the input
p_args = parser.parse_args()
result_dir = p_args.dir

result_dir = os.path.join('/data/seb/data', result_dir)
dest_dir = result_dir + "_img"
source_dir = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task999_7T/imagesTs'

if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

file_list = os.listdir(result_dir)
file_list = [x for x in file_list if x.endswith('gz')]
for sel_file in file_list:
    sel_file_name = hmisc.get_base_name(sel_file)
    sel_file_segm = os.path.join(result_dir, sel_file)
    sel_file_source_dir = [x for x in os.listdir(source_dir) if sel_file_name in hmisc.get_base_name(x)][0]
    sel_file_img = os.path.join(source_dir, sel_file_source_dir)
    A = nibabel.load(sel_file_segm).get_fdata()
    A = np.moveaxis(A, -1, 0)
    B = nibabel.load(sel_file_img).get_fdata()
    B = np.moveaxis(B, -1, 0)
    plot_obj = hplotc.ListPlot([A[::3], B[::3]], vmin=[(0,3), (0,1)])
    plot_obj.figure.savefig(os.path.join(dest_dir, sel_file_name + '.png'))
