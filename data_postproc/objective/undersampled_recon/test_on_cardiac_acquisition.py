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
Test stuff...
"""

dconfig = '/home/bugger/Documents/model_run/undersampled_recon/config_00'
recall_obj = recall.RecallUndersampled(dconfig, config_name='config_param.json')
recall_obj.mult_dict['config_00']['dir']['doutput'] = dconfig
modelrun_obj = recall_obj.get_model_object(recall_obj.mult_dict['config_00'])

"""
Define paths for CPX data
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

"""
Load the data...
"""

# Load cartesian ground truth..
# cpx_cartesian = scipy.io.loadmat(groundtruth_file)['reconstructed_data']
# cpx_cartesian = np.moveaxis(np.squeeze(cpx_cartesian), -1, 0)
cpx_cartesian = np.load(groundtruth_file)
cpx_cartesian = np.squeeze(cpx_cartesian)
hplotc.SlidingPlot(cpx_cartesian)

# Load fully sampled radial
cpx_obj = read_cpx.ReadCpx(radial_fs_file)
cpx_radial_fs = np.squeeze(cpx_obj.get_cpx_img())
# cpx_radial_fs = scipy.io.loadmat(radial_fs_file)['reconstructed_data']
# cpx_radial_fs = np.moveaxis(np.squeeze(cpx_radial_fs), -1, 0)
hplotc.SlidingPlot(cpx_radial_fs.sum(axis=0)[:, ::-1, ::-1])

# Load under sampled radial..
cpx_obj = read_cpx.ReadCpx(radial_us_file)
cpx_radial_us = cpx_obj.get_cpx_img()
cpx_radial_us = np.squeeze(cpx_radial_us)
hplotc.SlidingPlot(np.abs(cpx_radial_us).sum(axis=0)[:, ::-1, ::-1])

n_card = cpx_radial_us.shape[1]
image_to_gif_array = np.abs(cpx_radial_us).sum(axis=0)[:, ::-1, ::-1]
# hplotc.ListPlot(image_to_gif_array[0], cbar=True)
# image_to_gif_array[image_to_gif_array > 63000] = 63000
hmisc.convert_image_to_gif(image_to_gif_array,
                     output_path=os.path.join(output_path, 'radial_abs_sum.gif'),
                     n_slices=n_card,
                     nx=256, ny=256, duration=15 / (3 * n_card))

# Load non triggerd
cpx_obj = read_cpx.ReadCpx(radial_no_trigger_file)
cpx_no_trigger = cpx_obj.get_cpx_img()
cpx_no_trigger = np.squeeze(cpx_no_trigger)
hplotc.SlidingPlot(cpx_no_trigger.sum(axis=0)[:, ::-1, ::-1])

n_card = cpx_no_trigger.shape[1]
image_to_gif_array = np.abs(cpx_no_trigger.sum(axis=0)[:, ::-1, ::-1])
hplotc.ListPlot(image_to_gif_array[0], cbar=True)
image_to_gif_array[image_to_gif_array > 63000] = 63000
hmisc.convert_image_to_gif(image_to_gif_array,
                     output_path=os.path.join(output_path, 'no_trigger_acq.gif'),
                     n_slices=n_card,
                     nx=256, ny=256, duration=15 / n_card)

# ORDER IS
# (NUMBER OF COILS, NUMBER OF CARDIAC PHASES, NX, NY)

# Needed to get the order correct
# img_to_predict = np.swapaxes(cpx_radial_us, 0, 1)
img_to_predict = np.swapaxes(cpx_no_trigger, 0, 1)
result_card = []
interm_result = []
counter = 0
for i_card in img_to_predict[:, -8:]:
    print('Processing slice ', counter)
    counter += 1

    input_array = i_card[None]
    # input_array = input_array.sum(axis=1, keepdims=True)
    input_array = harray.scale_minmax(input_array, is_complex=True)
    n_y, n_x = input_array.shape[-2:]
    x_inputed = harray.to_stacked(input_array, cpx_type='cartesian', stack_ax=0)
    x_inputed = x_inputed.T.reshape((n_x, n_y, -1)).T
    A_tensor = torch.as_tensor(x_inputed[np.newaxis]).float()

    if modelrun_obj.trained_modelrun_obj is not None:
        with torch.no_grad():
            A_tensor = modelrun_obj.trained_modelrun_obj.model_obj(A_tensor)
        interm_result.append(A_tensor.detach().numpy())

    with torch.no_grad():
        if modelrun_obj.config_param['model']['model_choice'] == 'gan':
            output = modelrun_obj.generator(A_tensor)
        elif modelrun_obj.config_param['model']['model_choice'] == 'cyclegan':
            output = modelrun_obj.netG_A2B(A_tensor)
        else:
            output = modelrun_obj.model_obj(A_tensor)

    if output.shape[1] > 1:
        output_cpx = output.numpy()[0][0] + 1j * output.numpy()[0][1]
        # Here we take the ABS of the output..
        output_abs = np.abs(output_cpx)
    else:
        # Output is either abs or real.. any case.. it is fine..
        output_abs = output.numpy()[0][0]

    result_card.append(output_abs)

result_card = np.array(result_card)
result_card = result_card[:, ::-1, ::-1]
hplotc.SlidingPlot(result_card)


n_card = result_card.shape[0]
cpx_cartesian
hplotc.ListPlot([result_card[0], cpx_cartesian[0]], cbar=True, augm='np.abs')
postproc_result[postproc_result > 3] = 3
hmisc.convert_image_to_gif(postproc_result,
                     output_path=os.path.join(output_path, 'result_double_rocket_model_untriggered.gif'),
                     n_card=n_card,
                     nx=256, ny=256, duration=15/n_card)


"""
Load some other matlab data...

See if Christine's data can be recovered... LOL no
"""

dmodel = '/home/bugger/Documents/model_run/various_us_recon/resnet_cartesian_05'
recall_obj = recall.RecallUndersampled(dmodel, config_name='config_param.json')
recall_obj.mult_dict['config_00']['dir']['doutput'] = dmodel
modelrun_obj = recall_obj.get_model_object(recall_obj.mult_dict['config_00'])

ddata = '/media/bugger/Data/2021_10_27/V9_27139/recon_data.mat'
matlab_obj = scipy.io.loadmat(ddata)['recon_data']
card_array = np.squeeze(np.moveaxis(matlab_obj, -1, 0))

interm_result = []
for i_card in card_array:
    input_array = i_card
    # input_array = input_array.sum(axis=1, keepdims=True)
    input_array = harray.scale_minmax(input_array, is_complex=True)
    n_y, n_x = input_array.shape[-2:]
    A_tensor = torch.as_tensor(input_array[np.newaxis, np.newaxis]).float()
    with torch.no_grad():
        res = modelrun_obj.model_obj(A_tensor)
    interm_result.append(res.detach().numpy())

hplotc.SlidingPlot(np.array(interm_result))

"""
Aaand continue with reconstructed data from Bart
"""

import helper.reconstruction as hrecon
import sigpy.mri

dmodel = '/home/bugger/Documents/model_run/various_us_recon/resnet_cartesian_05'
recall_obj = recall.RecallUndersampled(dmodel, config_name='config_param.json')
recall_obj.mult_dict['config_00']['dir']['doutput'] = dmodel
modelrun_obj = recall_obj.get_model_object(recall_obj.mult_dict['config_00'])

# # #
# Now load the new data
# # #

dunsorted = '/media/bugger/WORK_USB/bart_data/bart_17_2_unsorted.mat'
unsorted_array = scipy.io.loadmat(dunsorted)['bart_17_2_unsorted']

sel_sin_file = '/media/bugger/WORK_USB/2021_12_01/ca_29045/ca_01122021_1019026_17_2_transverse_retro_radialV4.sin'
trajectory = hrecon.get_trajectory_sin_file(sel_sin_file)
ovs = float(hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_grid_overs_factor'))
width = int(hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_grid_kernel_size'))
n_coil = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_channel_names'))
n_card = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_cardiac_phases'))

result = []
for i_coil in range(n_coil):
    selected_data = unsorted_array[:, i_coil::n_coil][:, :trajectory.shape[0]]
    selected_data = np.moveaxis(selected_data, -1, 0)
    temp_img = sigpy.nufft_adjoint(selected_data, coord=trajectory)
    result.append(temp_img)

result = np.array(result)
hplotc.ListPlot(np.abs(result).sum(axis=0, keepdims=True), augm='np.abs')

result_tens = torch.from_numpy(np.sum(np.abs(result), axis=0))[None, None].float()

with torch.no_grad():
    res_model = modelrun_obj.model_obj(result_tens)

hplotc.ListPlot([res_model.numpy()[0][0], result_tens])

