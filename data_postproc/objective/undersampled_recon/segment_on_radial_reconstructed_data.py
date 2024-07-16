"""
Test out a segmentation model and a reconstruction
"""

import objective.undersampled_recon.executor_undersampled_recon as executor_radial
import objective.mm_segment.executor_mm_segment as executor_mm
import os

import matplotlib.pyplot as plt
import helper.plot_class as hplotc
import numpy as np
import torch
import helper.misc as hmisc
import helper.array_transf as harray
import os
import re
import reconstruction.ReadCpx as read_cpx
import nibabel as nib
import nibabel as nib
import skimage.transform as sktransf


def process_output(x):
    x_padded = np.concatenate([np.zeros(x.shape[-2:])[None], x.numpy()[0]])
    x_rounded = np.isclose(x_padded, 1, atol=0.8).astype(int)
    x_maxed = np.argmax(x_rounded, axis=0)  # .astype(int)
    return x_maxed


"""
Radial data
"""

# What happens when we put in an acquired image..?
output_path = '/home/bugger/Documents/model_run/various_us_recon'
patient_dir = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_13/V9_17069'

# Cpx file of carteisna one is of course... SENSE'd
transverse_groundtruth_file = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac/V9_17069/transverse/v9_13022021_1343308_5_3_cine1slicer2_traV4.npy'
transverse_radial_us_file = os.path.join(patient_dir, 'v9_13022021_1346189_7_3_transradialfast_high_timeV4.cpx')
cpx_transverse_cartesian = np.load(transverse_groundtruth_file)
cpx_transverse_cartesian = np.squeeze(cpx_transverse_cartesian)

ch4_groundtruth_file = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac/V9_17069/4ch/v9_13022021_1403154_15_3_4chV4.npy'
ch4_radial_fs_file = os.path.join(patient_dir, 'v9_13022021_1405014_16_3_4ch_radialV4.cpx')
ch4_radial_us_file = os.path.join(patient_dir, 'v9_13022021_1405556_17_3_4ch_radial_high_timeV4.cpx')

# Load it...
ch4_cartesian_array = np.load(ch4_groundtruth_file)
ch4_cartesian_array = np.squeeze(ch4_cartesian_array)

cpx_obj = read_cpx.ReadCpx(ch4_radial_us_file)
ch4_radial_us = cpx_obj.get_cpx_img()
ch4_radial_us = np.squeeze(ch4_radial_us)


cpx_obj = read_cpx.ReadCpx(ch4_radial_fs_file)
ch4_radial_fs = cpx_obj.get_cpx_img()
ch4_radial_fs = np.squeeze(ch4_radial_fs)


# Find 4Ch thing..
# Or SA thnig
"""
Radial model
"""

model_path = '/home/bugger/Documents/model_run/various_us_recon/resnet_radial_20'
config_param = hmisc.convert_remote2local_dict(model_path, path_prefix='/media/bugger/MyBook/data/semireal')
config_param['data']['batch_size'] = 1
decision_obj = executor_radial.DecisionMakerRecon(config_file=config_param, debug=True, load_model_only=True, inference=False, device='cpu')  # ==>>
modelrun_obj = decision_obj.decision_maker()
modelrun_obj.load_weights()
if modelrun_obj.model_obj:
    modelrun_obj.model_obj.eval()
else:
    modelrun_obj.generator.eval()

"""
Segmentation model
"""

i_model_path = '/home/bugger/Documents/model_run/mm_segment/unet_n4itk_dice_haus'
config_param = hmisc.convert_remote2local_dict(i_model_path, path_prefix='/media/bugger/MyBook/data/semireal')
config_param['data']['batch_size'] = 1
config_param['dir']['ddata'] = '/media/bugger/MyBook/data/m&m/MnM_dataset'

decision_obj = executor_mm.DecisionMakerMMSegment(config_file=config_param, debug=False,
                                               load_model_only=True, inference=True, device='cpu')  # ==>>
modelrun_obj_segm = decision_obj.decision_maker()
modelrun_obj_segm.load_weights()
if modelrun_obj_segm.model_obj:
    modelrun_obj_segm.model_obj.eval()
else:
    modelrun_obj_segm.generator.eval()


"""
Segment cartesian acquisition
"""

sel_card = 15
input_array_cart = np.abs(ch4_cartesian_array[sel_card])
input_array_cart = harray.scale_minmax(input_array_cart)
A_tensor = torch.as_tensor(input_array_cart[None, None]).float()

with torch.no_grad():
    if modelrun_obj_segm.config_param['model']['model_choice'] == 'gan':
        output_segm_cartesian = modelrun_obj_segm.generator(A_tensor)
    elif modelrun_obj_segm.config_param['model']['model_choice'] == 'cyclegan':
        output_segm_cartesian = modelrun_obj_segm.netG_A2B(A_tensor)
    else:
        output_segm_cartesian = modelrun_obj_segm.model_obj(A_tensor)

output_segm_cartesian_proc = process_output(output_segm_cartesian)
hplotc.ListPlot([output_segm_cartesian_proc, input_array_cart])

"""
Segment radial recovered acquisition
"""

sel_card = 3
input_array_radial = ch4_radial_fs[-8:, sel_card:(sel_card+1)]
input_array_radial = np.swapaxes(input_array_radial, 1, 0)[:, :, ::-1, ::-1]
input_array_radial = harray.scale_minmax(input_array_radial, is_complex=True)

n_y, n_x = input_array_radial.shape[-2:]
x_inputed = harray.to_stacked(input_array_radial, cpx_type='cartesian', stack_ax=0)
x_inputed = x_inputed.T.reshape((n_x, n_y, -1)).T
A_tensor = torch.as_tensor(x_inputed[np.newaxis]).float()

with torch.no_grad():
    if modelrun_obj.config_param['model']['model_choice'] == 'gan':
        output_radial = modelrun_obj.generator(A_tensor)
    elif modelrun_obj.config_param['model']['model_choice'] == 'cyclegan':
        output_radial = modelrun_obj.netG_A2B(A_tensor)
    else:
        output_radial = modelrun_obj.model_obj(A_tensor)

output_radial = harray.scale_minmax(output_radial.numpy())
A_tensor = torch.as_tensor(output_radial).float()

with torch.no_grad():
    if modelrun_obj_segm.config_param['model']['model_choice'] == 'gan':
        output_segm_radial = modelrun_obj_segm.generator(A_tensor)
    elif modelrun_obj_segm.config_param['model']['model_choice'] == 'cyclegan':
        output_segm_radial = modelrun_obj_segm.netG_A2B(A_tensor)
    else:
        output_segm_radial = modelrun_obj_segm.model_obj(A_tensor)

output_segm_radial_proc = process_output(output_segm_radial)


"""
Segment radial originial acquisition
"""

segm_radial_input = np.abs(input_array_radial).sum(axis=1, keepdims=True)
A_tensor = torch.as_tensor(segm_radial_input).float()

with torch.no_grad():
    if modelrun_obj_segm.config_param['model']['model_choice'] == 'gan':
        output_segm_radial_orig = modelrun_obj_segm.generator(A_tensor)
    elif modelrun_obj_segm.config_param['model']['model_choice'] == 'cyclegan':
        output_segm_radial_orig = modelrun_obj_segm.netG_A2B(A_tensor)
    else:
        output_segm_radial_orig = modelrun_obj_segm.model_obj(A_tensor)

output_segm_radial_orig_proc = process_output(output_segm_radial_orig)

hplotc.ListPlot([segm_radial_input, output_segm_radial_orig_proc])

hplotc.ListPlot([[segm_radial_input[0][0], output_segm_radial_orig_proc], [output_radial[0][0], output_segm_radial_proc], [np.abs(ch4_cartesian_array[sel_card]), output_segm_cartesian_proc]])
