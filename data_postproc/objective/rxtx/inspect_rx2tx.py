# encoding: utf-8

import getpass
import os
import sys


"""
Determine if we are working remote or local.
Add a project path for proper loading of packages
"""

# Deciding which OS is being used
if getpass.getuser() == 'bugger':
    local_system = True
    manual_mode = True
else:
    import matplotlib as mpl
    mpl.use('Agg')  # Hopefully this makes sure that we can plot/save stuff
    local_system = False
    manual_mode = False

if local_system:
    project_path = "/"
    model_path = os.path.join(project_path, 'config_template')
else:
    project_path = "/home/seb/code/pytorch_in_mri"
    model_path = "/home/seb/code/pytorch_in_mri/config_template"


print('Adding to path: ', project_path)
sys.path.append(project_path)

import torch.nn.functional as F
import helper.plot_class as hplotc
import numpy as np
import json
import importlib
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.misc as hmisc
import helper.model_setting as hmodel_set
import torch.utils.data
import objective.rx2tx.executor_rx2tx as executor


"""
Load model object with model path
"""

load_model_path = False
load_with_file = True
load_model_only = False

if load_model_path:
    config_file_name = "inhomog_removal.json"  # ==>>
    config_file = os.path.join(model_path, config_file_name)

    with open(config_file, 'r') as f:
        text_obj = f.read()
        config_model = json.loads(text_obj)

    if local_system:
        # With this we store the data in the project directory
        config_model['dir']['dtemplate'] = model_path
        config_model['dir']['doutput'] = "/home/bugger/Documents/model_run/test_results"
        config_model['dir']['ddata'] = "/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels"  # ==>>
    else:
        config_model['dir']['doutput'] = "/data/seb/model_run/test_run"
        config_model['dir']['dtemplate'] = model_path
        config_model['dir']['ddata'] = "/home/seb/data/b1shimsurv_all_channels"  # ==>>

    config_model['model']['n_epoch'] = 500


    mult_dict = hmodel_set.create_mult_dict(config_model, **config_model, debug=False)
    model_path_list = hmodel_set.create_config_dir(config_model['dir']['doutput'], mult_dict, debug=False)

    full_model_path = model_path_list[0]
    # This should be the new way.. to distinguish between GANs and non GANs
    decision_obj = executor.DecisionMaker(model_path=full_model_path, debug=True, index_gpu=None)  # ==>>

elif load_with_file:
    """
    Load model with config param.
    """

    # GAN
    # dir_path = '/home/bugger/Documents/model_run/test_run_gan/config_00'
    # config_param = hmisc.convert_remote2local_dict(dir_path)

    # Xnet
    config_path = '/home/bugger/Documents/model_run/config_xnet_flavio'
    config_path = '/home/bugger/Documents/model_run/rxtx'
    config_path = '/home/bugger/Documents/model_run/rxtx_double_2'
    config_param = hmisc.convert_remote2local_dict(config_path, path_prefix='/home/bugger/Documents/data')
    # ddata = '/home/bugger/Documents/data/simulation/cardiac/rxtx'
    ddata = '/home/bugger/Documents/data/simulation/prostate_mri_mrl'
    config_param['dir']['ddata'] = ddata
    decision_obj = executor.DecisionMaker(config_file=config_param, debug=True)  # ==>>

elif load_model_only:
    # Yeah/// define some more stuff here...
    pass

"""
Load the model..
"""

# Recreate the modelling object
A = decision_obj.decision_maker()
# A.load_weights()

"""
Here we perform the double model thing
"""
dir_weights_second = os.path.join(A.config_param['dir']['doutput'], 'second_' + A.name_model_weights)
dir_weights = os.path.join(A.config_param['dir']['doutput'], A.name_model_weights)

weight_model = torch.load(dir_weights, map_location=A.device)
weight_model_second = torch.load(dir_weights_second, map_location=A.device)

A.model_obj.load_state_dict(weight_model)
A.model_obj_second.load_state_dict(weight_model_second)

with torch.no_grad():  # IMPORTANT
    for container in A.test_loader:
        X, y, mask = container

        X_real = X[:, ::2]
        y_real = y[:, ::2]
        X_imag = X[:, 1::2]
        y_imag = y[:, 1::2]

        with torch.no_grad():
            torch_pred = A.model_obj(X_real)
        with torch.no_grad():
            torch_pred_second = A.model_obj_second(torch_pred)

        hplotf.plot_3d_list(torch_pred_second.numpy())
        res = torch_pred.numpy() / np.max(torch_pred.numpy()) + 1j * torch_pred_second.numpy() / np.max(torch_pred_second.numpy())
        target = y_real.numpy() + 1j * y_imag.numpy()
        hplotf.plot_3d_list(np.stack([target, res], axis=1), augm='np.real')
        hplotf.plot_3d_list([target, res], augm='np.angle')
        hplotf.plot_3d_list([target, res], augm='np.imag')
        hplotf.plot_3d_list([target, res], augm='np.imag')

"""
Recalculate loss, and display error
"""

l1_test_loss = torch.nn.L1Loss()
mse_test_loss = torch.nn.MSELoss()
kl_test_loss = torch.nn.KLDivLoss()
print('l1', l1_test_loss(torch.as_tensor(y_pred0), torch.as_tensor(y0)))
print('mse', mse_test_loss(torch.as_tensor(y_pred0), torch.as_tensor(y0)))
print('KL', kl_test_loss(torch.as_tensor(y_pred0), torch.as_tensor(y0)))
print('Own loss', A.loss_obj(torch.as_tensor(y_pred0), torch.as_tensor(y0)))

hplotc.SlidingPlot(y_pred0-y0, ax_3d=True)
hplotc.SlidingPlot(y_pred0-y0, ax_3d=False)
hplotf.plot_3d_list(y0, augm='np.abs')
hplotf.plot_3d_list(y_pred0, augm='np.abs')

hplotf.plot_3d_list(np.concatenate([x0[np.newaxis], y0[np.newaxis], y_pred0[np.newaxis]], axis=0)[:,0], augm='np.abs')

"""
Display results on a coil by coil basis
"""

c2, c1, _, _ = x0.shape
# Look at the images coil per coil.. is a bit more easy..
for i_augm in plot_augmentation:
    for i_item in range(c2):
        for i_coil in range(c1):
            plot_array = np.stack([x0[i_item, i_coil], y0[i_item, i_coil], y_pred0[i_item, i_coil]], axis=0)[np.newaxis]
            fig_handle = hplotf.plot_3d_list(plot_array, figsize=(15, 10), dpi=75, augm=i_augm,
                                             title='batch nr {} coil nr {}'.format(str(i_item), str(i_coil)))

"""
Show how off we are in a line graph.. 
"""

plt.close('all')
sel_batch = 0
for sel_coil in range(8):
    for sel_line in range(0, 256, 128):
        plt.figure()
        plt.plot(x0[sel_batch, sel_coil, :, sel_line], 'r', label='input')
        plt.plot(y0[sel_batch, sel_coil, :, sel_line], 'b', label='target')
        plt.plot(y_pred0[sel_batch, sel_coil, :, sel_line], 'k', label='pred')
        plt.legend()

"""
Inspect intermediate layer results.. The code below is subject to a lot of changes...
"""

def get_model_up_parameters(model_up, sel_coil, sel_weight):
    # Here we are going to look into a specific model at a specific time..
    list_weights = list(model_up[sel_coil].children())
    n_weights = len(list_weights)
    print(f'Amount of weights... {n_weights}')
    list_parameters = list(list_weights[sel_weight].parameters())
    n_parameters = len(list_parameters)
    print(f'Amount of parameters... {n_parameters}')
    return list_parameters

res_0 = [get_model_up_parameters(A.generator.model_up, x, 0) for x in range(8)]
first_param = [x[0].detach().numpy().reshape(-1, 3, 3)[0:5] for x in res_0]
len(first_param)
[x.shape for x in first_param]
hplotf.plot_3d_list(first_param)
# # # #


# # # #


# Check model layers... - This is again... outdated...
model_choice = A.config_param['model']['model_choice']
if model_choice == "xnet":

    # # Testing cold model creation
    import model.XNet as model_xnet
    import helper_torch.misc as htmisc

    model_obj = model_xnet.XNet(debug=True)
    model_obj = A.model_obj
    model_obj = A.generator
    res = htmisc.get_all_children(model_obj)
    import helper.array_transf as harray
    x1 = harray.to_stacked(x0)
    x2 = np.moveaxis(x1[0], -1, 0)
    input_tensor_list = torch.split(torch.as_tensor(x2), 2, dim=1)

    # Down layer
    with torch.no_grad():
        down_results = [model_obj.mod_down(x) for x in input_tensor_list]
        ftr_inp, down_stack = zip(*down_results)
        plot_mid = [np.squeeze(x.numpy()) for x in ftr_inp]

    len(down_results[0][0])
    hplotf.plot_3d_list(down_results[0][0][:, ::10][np.newaxis], ax_off=True)

    # Feature layer
    with torch.no_grad():
        cat_result_down = torch.cat(ftr_inp, dim=1)
        cat_result_down_perm = cat_result_down.permute((0, 2, 3, 1))

        result_mid_perm = model_obj.mod_mid(cat_result_down_perm)
        result_mid = result_mid_perm.permute((0, 3, 1, 2))
        n_concat = result_mid.shape[1]
        result_mid_split = torch.split(result_mid, n_concat//8, dim=1)

    # Up sample layer..
    with torch.no_grad():
        result_up = [model_obj.mod_up(x, stack=y) for x, y in zip(result_mid_split, down_stack)]  # Model...
        output = torch.cat(result_up, dim=1)

    # Display the lowest feature space and the original image
    for i in range(len(plot_mid)):
        hplotf.plot_3d_list([ plot_mid[i][:, 0:16]], ax_off=True, augm='np.abs', vmin=(0, 5), cbar=True)

    # Display the intermediate output layers..
    for i_coil in range(len(down_stack)):
        sel_coil = down_stack[i_coil]
        for i_depth in range(len(sel_coil)):
            hplotf.plot_3d_list(sel_coil[i_depth])

    # Display the feature middel layer
    for i in result_mid_split:
        hplotf.plot_3d_list(i, ax_off=True)

    # Display the final output..
    hplotf.plot_3d_list(result_up)

    """Deeper into model up..."""
    for sel_coil in range(8):
        stack = list(down_stack[sel_coil])
        output = result_mid_split[sel_coil]
        interm_output = []
        with torch.no_grad():
            for i, layer in enumerate(mod_up.up_sample_layers):
                downsample_layer = stack.pop()
                layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
                output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
                output = torch.cat([output, downsample_layer], dim=1)
                output = layer(output)
                interm_output.append(output)

            output = mod_up.conv2(output)
            interm_output.append(output)

            if mod_up.output_activation is not None:
                output = mod_up.output_actv_layer(output)
                interm_output.append(output)

        # Display the output..
        hplotf.plot_3d_list([np.squeeze(x) for x in interm_output], ax_off=True)

    if A.config_param['data']['fourier_transform']:
        # See if the predicted real part is even useful...
        import data_generator.Rx2Tx as data_gen
        import helper.array_transf as harray
        derp = data_gen.DataSetSurvey2B1_all(ddata=A.config_param['dir']['ddata'], input_shape=(16, 512, 256),
                                      fourier_transform=True, transform_type='complex')
        a, b = derp.__getitem__(0)
        a_np = a.numpy()
        a_real = a_np[:8]
        a_imag = a_np[8:]
        with torch.no_grad():
            a_pred_real = A.model_obj(a[:8][np.newaxis]).numpy()

        a_fft = harray.transform_kspace_to_image_fftn(a_real + 1j * a_imag, dim=(-2, -1))
        a_pred_fft = harray.transform_kspace_to_image_fftn(a_pred_real[0] - np.mean(a_pred_real) + 1j * a_imag, dim=(-2, -1))
        hplotf.plot_3d_list(a[np.newaxis])
        hplotf.plot_3d_list(a_fft[np.newaxis], augm='np.abs')
        hplotf.plot_3d_list(a_pred_fft[np.newaxis], augm='np.abs')


"""
Investigate losses etc
"""

from objective.recall_base import RecallBase
import json

recall_obj = RecallBase(config_run_file=A.config_param)
with open(os.path.join(recall_obj.model_path, recall_obj.name_model_hist), 'r') as f:
    text_obj = f.read()
    history_obj = json.loads(text_obj)

train_loss = history_obj['train_loss']
val_loss = history_obj['val_loss']
