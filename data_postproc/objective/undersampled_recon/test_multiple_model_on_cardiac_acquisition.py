import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch
import helper.plot_class as hplotc
import helper.misc as hmisc
import objective.undersampled_recon.executor_undersampled_recon as executor
import objective.undersampled_recon.recall_undersampled_recon as recall
import helper.array_transf as harray
import reconstruction.ReadCpx as read_cpx
import reconstruction.ReadRec as read_rec


"""
I want to see if we use the opinion of multiple models..

Can we make it better...?
"""


patient_dir = '/media/bugger/MyBook/data/7T_scan/cardiac'
for d, _, f in os.walk(patient_dir):
    if len(f):
        filter_f = [x for x in f if 'transradialfastV4' in x]

output_path = '/home/bugger/Documents/presentaties/RF_meetings/RF_meeting_20210915'
patient_dir = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_13/V9_17069'

# Cpx file of carteisna one is of course... SENSE'd
groundtruth_file = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac/V9_17069/transverse/v9_13022021_1343308_5_3_cine1slicer2_traV4.npy'
radial_fs_file = os.path.join(patient_dir, 'v9_13022021_1345094_6_3_transradialfastV4.cpx')
radial_us_file = os.path.join(patient_dir, 'v9_13022021_1346189_7_3_transradialfast_high_timeV4.cpx')
radial_no_trigger_file = os.path.join(patient_dir, 'v9_13022021_1347235_8_3_transradial_no_trigV4.cpx')


# Load non triggerd
cpx_obj = read_cpx.ReadCpx(radial_no_trigger_file)
cpx_no_trigger = cpx_obj.get_cpx_img()
cpx_no_trigger = np.squeeze(cpx_no_trigger)
hplotc.SlidingPlot(cpx_no_trigger.sum(axis=0)[:, ::-1, ::-1])

# Select one slice...
sel_card = 40
input_array = np.swapaxes(cpx_no_trigger[:, sel_card:(sel_card+1), ::-1, ::-1], 0, 1)
input_array = np.abs(input_array).sum(axis=1, keepdims=True)
# input_array = input_array.sum(axis=1, keepdims=True)
# input_array = harray.scale_minmax(input_array, is_complex=True)
# n_y, n_x = input_array.shape[-2:]
# x_inputed = harray.to_stacked(input_array, cpx_type='cartesian', stack_ax=0)
# x_inputed = x_inputed.T.reshape((n_x, n_y, -1)).T
import skimage.transform as sktransf
input_array = sktransf.resize(input_array[0][0], (256, 256), anti_aliasing=False)
A_tensor = torch.from_numpy(input_array[None, None]).float()
A_tensor.shape
hplotc.ListPlot(A_tensor)

x_size, y_size = A_tensor.shape[-2:]
x_range = np.linspace(-x_size//2, x_size//2, x_size)
y_range = np.linspace(-y_size // 2, y_size // 2, y_size)
X, Y = np.meshgrid(x_range, y_range)
mask_array = np.sqrt(X ** 2 + Y ** 2) <= x_size//2
mask_tensor = torch.from_numpy(mask_array[None, None])

"""
Test stuff...
"""

dconfig_dir = '/home/bugger/Documents/model_run/undersampled_recon_2022'


result_card = []
counter = 0
for i_config in sorted(os.listdir(dconfig_dir)):
    print('Processing slice ', i_config)
    dconfig = os.path.join(dconfig_dir, i_config)
    recall_obj = recall.RecallUndersampled(dconfig, config_name='config_param.json')
    recall_obj.mult_dict['config_00']['dir']['doutput'] = dconfig
    modelrun_obj = recall_obj.get_model_object(recall_obj.mult_dict['config_00'])

    with torch.no_grad():
        if modelrun_obj.config_param['model']['model_choice'] == 'gan':
            output = modelrun_obj.generator(A_tensor)
        elif modelrun_obj.config_param['model']['model_choice'] == 'cyclegan':
            output = modelrun_obj.netG_A2B(A_tensor)
        else:
            output = modelrun_obj.model_obj(A_tensor * mask_tensor)

    if output.shape[1] > 1:
        output_cpx = output.numpy()[0][0] + 1j * output.numpy()[0][1]
        # Here we take the ABS of the output..
        output_abs = np.abs(output_cpx)
    else:
        # Output is either abs or real.. any case.. it is fine..
        output_abs = output.numpy()[0][0]

    result_card.append(output_abs)

result_card = np.array(result_card)
hplotc.ListPlot(input_array)
hplotc.SlidingPlot(harray.scale_minmax(result_card, axis=(-2,-1)))
hplotc.ListPlot(np.mean(result_card, axis=0))