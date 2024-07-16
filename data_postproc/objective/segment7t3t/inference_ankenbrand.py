import torch
import numpy as np
import helper.plot_class as hplotc
import matplotlib.pyplot as plt
import torch
import fastai
import nibabel
import sys
import helper.misc as hmisc
import argparse

from fastai.vision.all import Tensor, load_learner
from objective_configuration.segment7T3T import dankebrand_pkl, get_path_dict
import os

"""
Model from https://github.com/chfc-cmi/cmr-seg-tl/releases
"""

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str)
# Allows for a selection of models
p_args = parser.parse_args()
dataset = p_args.dataset

path_dict = get_path_dict(dataset)
dimg = path_dict['dimg']
dresults = os.path.join(path_dict['dresults'], 'ankebrand')
if not os.path.isdir(dresults):
    os.makedirs(dresults)

"""
Load some funcs... Needed to load the pickle object
"""

def label_func(x):
    return str(x['file']).replace("images", "../../data/data/masks_2class")


def acc_seg(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()


def multi_dice(input:Tensor, targs:Tensor, class_id=0, inverse=False):
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    # replace all with class_id with 1 all else with 0 to have binary case
    output = (input == class_id).float()
    # same for targs
    targs = (targs.view(n, -1) == class_id).float()
    if inverse:
        output = 1 - output
        targs = 1 - targs
    intersect = (output * targs).sum(dim=1).float()
    union = (output+targs).sum(dim=1).float()
    res = 2. * intersect / union
    res[torch.isnan(res)] = 1
    return res.mean()


def diceComb(input:Tensor, targs:Tensor):
    return multi_dice(input, targs, class_id=0, inverse=True)


def diceLV(input:Tensor, targs:Tensor):
    return multi_dice(input, targs, class_id=1)


def diceMY(input:Tensor, targs:Tensor):
    return multi_dice(input, targs, class_id=2)


trainedModel = load_learner(dankebrand_pkl)

"""
Load some data
"""

import helper.array_transf as harray
from skimage.util import img_as_ubyte, img_as_uint, img_as_int
dimg_files = os.listdir(dimg)
for i_file in dimg_files:
    dest_file = os.path.join(dresults, i_file)
    sel_file = os.path.join(dimg, i_file)
    temp_array = hmisc.load_array(sel_file).T[:, ::-1, ::-1]
    print(i_file, dest_file)
    print('Shape of files ', temp_array.shape)
    if temp_array.ndim == 2:
        temp_array = temp_array[:, :, None]

    # Loop over all the locations..
    pred_array = []
    for i_array in temp_array:
        i_array = img_as_ubyte(harray.scale_minmax(i_array))
        # Try four different rotations and find the best predictor
        rotated_predictions = []
        for k_rot in range(4):
            tens_array = torch.from_numpy(np.rot90(i_array, k=k_rot, axes=(-2, -1))[None].copy()).float()
            testDL = trainedModel.dls.test_dl(tens_array)
            predictions, _ = trainedModel.get_preds(dl=testDL)
            predictions = predictions.argmax(dim=1)
            n_voxels = (predictions > 0).sum()
            rotated_predictions.append((predictions, n_voxels, k_rot))

        # Select the best predictor by sorting based on the amount of voxels predicted
        # Then rotate it back to keep alignment with the labels
        sorted_rotated_predictions = sorted(rotated_predictions, key=lambda x: x[1])
        rot_pred, _, sel_rot = sorted_rotated_predictions[-1]
        predictions = np.rot90(rot_pred, k=-sel_rot, axes=(-2, -1))

        # Give the class one preditions the value 3
        class_one_ind = predictions == 1
        predictions[class_one_ind] = 3
        pred_array.append(predictions)
    pred_array = np.concatenate(pred_array)
    print('\tShape pred', pred_array.shape)
    nibabel_obj = nibabel.Nifti1Image(pred_array.T[::-1, ::-1, :], np.eye(4))
    nibabel.save(nibabel_obj, dest_file)
