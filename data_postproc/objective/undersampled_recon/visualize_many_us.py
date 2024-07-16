"""
Lets make here a scripts that shows the result when doing

Undersampled  (radially) Radial -> Fully sampled radial
"""

import os
import numpy as np
import torch
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.misc as hmisc
import objective.undersampled_recon.executor_undersampled_recon as executor
import helper.array_transf as harray
from skimage.util import img_as_ubyte, img_as_uint
import skimage.transform as sktransf
import imageio
import reconstruction.ReadCpx as read_cpx
import reconstruction.ReadRec as read_rec


""" 
Load data on which we are going to test stuff
"""

# What happens when we put in an acquired image..?
output_path = '/home/bugger/Documents/model_run/various_us_recon'
# output_path = '/home/bugger/Documents/model_run/various_us_recon_radial_only'
patient_dir = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_13/V9_17069'

# Cpx file of carteisna one is of course... SENSE'd
groundtruth_file = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac/V9_17069/transverse/v9_13022021_1343308_5_3_cine1slicer2_traV4.npy'
radial_fs_file = os.path.join(patient_dir, 'v9_13022021_1345094_6_3_transradialfastV4.cpx')
radial_us_file = os.path.join(patient_dir, 'v9_13022021_1346189_7_3_transradialfast_high_timeV4.cpx')
# radial_us_file = os.path.join(patient_dir, 'v9_13022021_1350388_11_3_p2ch_radial_high_timeV4.cpx')
# radial_us_file = os.path.join(patient_dir, 'v9_13022021_1405556_17_3_4ch_radial_high_timeV4.cpx')
radial_no_trigger_file = os.path.join(patient_dir, 'v9_13022021_1347235_8_3_transradial_no_trigV4.cpx')
cpx_cartesian = np.load(groundtruth_file)
cpx_cartesian = np.squeeze(cpx_cartesian)
hplotc.SlidingPlot(cpx_cartesian)

# Load under sampled radial..
cpx_obj = read_cpx.ReadCpx(radial_us_file)
cpx_radial_us = cpx_obj.get_cpx_img()
cpx_radial_us = np.squeeze(cpx_radial_us)
hplotc.SlidingPlot(cpx_radial_us.sum(axis=0)[:, ::-1, ::-1])

img_to_predict = np.swapaxes(cpx_radial_us, 0, 1)#[:, :, ::-1, ::-1]

"""
Load a model config
"""

# COnfig location of the current model...
dconfig = '/home/bugger/Documents/model_run/various_us_recon'
# dconfig = '/home/bugger/Documents/model_run/various_us_recon_radial_only'
config_wide_result = []
label_wide_result = []
for i_config in sorted([x for x in os.listdir(dconfig) if os.path.isdir(os.path.join(dconfig, x))]):
    print(i_config)
    # if i_config == 'resnet_radial_80':
    #     break
    if i_config == 'resnet_cartesian_80':
        break
    model_path = os.path.join(dconfig, i_config)

    config_param = hmisc.convert_remote2local_dict(model_path, path_prefix='/media/bugger/MyBook/data/semireal')
    config_param['data']['batch_size'] = 1

    # if config_param['data'].get('trained_model_config', {}).get('status', False):
    #     local_pretrained_path = '/home/bugger/Documents/model_run/undersampled_recon/pretrained_model/config_00'
    #     config_param['data']['trained_model_config']['model_path'] = local_pretrained_path

    decision_obj = executor.DecisionMakerRecon(config_file=config_param, debug=True,
                                               load_model_only=True, inference=False, device='cpu')  # ==>>

    modelrun_obj = decision_obj.decision_maker()
    modelrun_obj.load_weights()

    if modelrun_obj.model_obj:
        modelrun_obj.model_obj.eval()
    else:
        modelrun_obj.generator.eval()

    # if modelrun_obj.pretrained_modelrun_obj is not None:
    #     print('Pretrained model exists')
    #     modelrun_obj.pretrained_modelrun_obj.model_obj.eval()

    result_card = []
    interm_result = []
    counter = 0
    # We do only one cardiac phase for now to speed up comparisson...
    print('')
    for i_card in img_to_predict[:, -8:]:
        print('Processing slice ', counter, end='\r')
        counter += 1

        input_array = i_card[None]

        if 'cartesian' in i_config:
            input_array = np.abs(input_array).sum(axis=1, keepdims=True)
            input_array = harray.scale_minmax(input_array, is_complex=False)
            A_tensor = torch.as_tensor(input_array).float()
        else:
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
    config_wide_result.append(result_card)
    label_wide_result.append(i_config)


for ilabel, icardiac in zip(label_wide_result, config_wide_result):
    print(ilabel)
    print(icardiac.shape)
    n_card = icardiac.shape[0]
    hmisc.convert_image_to_gif(icardiac,
                         output_path=os.path.join(output_path, f'{ilabel}_transverse.gif'),
                         n_card=n_card,
                         nx=256, ny=256, duration=10/n_card)