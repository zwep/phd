# encoding: utf-8

import os
import importlib
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch

import helper.plot_class as hplotc
import helper.plot_fun as hplotf
import helper.misc as hmisc

import objective.unet_validation.executor_unet_validation as executor
import helper_torch.misc as htmisc


"""
Check if we can easily recover a model....
"""

path_with_configs = '/home/bugger/Documents/model_run/unet3d_ex'
i_config = os.listdir(path_with_configs)[0]
print(i_config)
# dir_path = '/home/bugger/Documents/model_run/test_run/config_02'
dir_path = os.path.join(path_with_configs, i_config)
dir_path = '/home/bugger/Documents/model_run/config_00_unet_validation_3'
dir_path = '/home/bugger/Documents/model_run/config_unet_temp'

config_param = hmisc.convert_remote2local_dict(dir_path, path_prefix='/home/bugger/Documents/data/grand_challenge')


"""
Load the model..
"""
importlib.reload(executor)
# Recreate the modelling object
A = executor.ExecutorUnetValidation(config_file=config_param, debug=True)
A.load_weights()

# Load a test example WITH something in its target value..
for a, b in A.test_loader:
    print(b.sum())
    if b.sum() > 0:
        break

hplotc.SlidingPlot(np.moveaxis(b.numpy(), -1, 0))
hplotc.SlidingPlot(np.moveaxis(a.numpy(), -1, 0))

# Some memory issues..
A.model_obj.eval()  # IMPORTANT
with torch.no_grad():  # IMPORTANT
    c = A.model_obj(a)

hplotc.SlidingPlot(np.moveaxis(c.numpy(), -1, 0))
hplotc.SlidingPlot(np.moveaxis(b.numpy(), -1, 0))
hplotc.SlidingPlot(np.moveaxis(a.numpy(), -1, 0))

loss_name = A.config_param['model']['loss']
loss_obj = A.loss_dict.get(loss_name, None)
temp_loss = loss_obj(ignore_index=0)

res_loss = A.loss_obj(c, b[np.newaxis])
res_loss = temp_loss(c, b[np.newaxis])


# Load specific example
import nibabel as nib
import re
t_center = 0
name_file = '133_image.nii.gz'
i_file = os.path.join(A.test_loader.dataset.input_dir, name_file)
x = nib.load(i_file).get_fdata()
i_file = os.path.join(A.test_loader.dataset.target_dir, name_file)
i_file = re.sub('image', 'mask', i_file)
y = nib.load(i_file).get_fdata()
hplotc.SlidingPlot(np.moveaxis(y, -1, 0))
hplotc.SlidingPlot(np.moveaxis(y, -1, 0)[(t_center-14):(t_center+14)])


# Look at the children of the model...
res = htmisc.get_all_children(A.model_obj)
for i in res:
    print(i)

# Evaluate the model on its intermediate layers
n = 3
with torch.no_grad():
    x = a[np.newaxis]
    for i_layer in res[:n]:
        x = i_layer(x)

hplotc.SlidingPlot(x)

# Run on a subpart of the model...
with torch.no_grad():
    res = torch.nn.Sequential(*list(list(A.model_obj.children())[0].children()))(a[np.newaxis])
