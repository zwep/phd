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
    project_path = "/home/bugger/PycharmProjects/pytorch_in_mri"
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
import objective.undersampled_recon.executor_undersampled_recon as executor



"""
Load model object with model path
"""

load_model_path = False
load_with_file = False
load_model_only = True

if load_model_path:
    config_file_name = "undersampled_recon.json"  # ==>>
    config_file = os.path.join(model_path, config_file_name)

    with open(config_file, 'r') as f:
        text_obj = f.read()
        config_model = json.loads(text_obj)

    if local_system:
        # With this we store the data in the project directory
        config_model['dir']['dtemplate'] = model_path
        config_model['dir']['doutput'] = "/home/bugger/Documents/model_run/test_results"
        config_model['dir']['ddata'] = "/home/bugger/Documents/data/semireal/prostate_simulation"  # ==>>
    else:
        config_model['dir']['doutput'] = "/data/seb/model_run/test_run"
        config_model['dir']['dtemplate'] = model_path
        config_model['dir']['ddata'] = "/home/seb/data/b1shimsurv_all_channels"  # ==>>

    config_model['model']['n_epoch'] = 1


    mult_dict = hmodel_set.create_mult_dict(config_model, **config_model, debug=False)
    model_path_list = hmodel_set.create_config_dir(config_model['dir']['doutput'], mult_dict, debug=False)

    full_model_path = model_path_list[0]
    # This should be the new way.. to distinguish between GANs and non GANs
    decision_obj = executor.DecisionMakerRecon(model_path=full_model_path, debug=True, index_gpu=None)  # ==>>

elif load_with_file:
    """
    Load model with config param.
    """
    # GAN
    # dir_path = '/home/bugger/Documents/model_run/test_run_gan/config_00'
    # config_param = hmisc.convert_remote2local_dict(dir_path)

    config_path = '/home/bugger/Documents/model_run/undersampling_results/undersampled_15'
    config_path = '/home/bugger/Documents/model_run/undersampling_results/undersampled_15_ynet'
    config_param = hmisc.convert_remote2local_dict(config_path, path_prefix='/home/bugger/Documents/data/semireal')
    decision_obj = executor.DecisionMaker(config_file=config_param, debug=True, inference=True)  # ==>>

elif load_model_only:
    model_path = '/home/bugger/Documents/model_run/undersampling_results/undersampled_15_noynet'
    model_path = '/home/bugger/Documents/model_run/undersampling_results/undersampled_15_ynet'
    config_param = hmisc.convert_remote2local_dict(model_path)

    import helper_torch.misc as htmisc

    gan_ind = False
    model_choice = config_param['model']['model_choice'].lower()
    if model_choice == 'gan':
        gan_ind = True
        model_choice = config_param['model']['config_gan']['generator_choice'].lower()

    print('You have chosen the following model object ', model_choice)
    if model_choice == 'resnet':
        # Get norm layer....
        config_resnet = config_param['model']['config_gan']['config_resnet']
        norm_layer_name = config_resnet.get('normalization_layer', 'InstanceNorm2d')  # InstanceNorm2d
        norm_layer = htmisc.module_selector(module_name=norm_layer_name)
        import functools
        if 'instance' in norm_layer_name.lower():
            norm_layer = functools.partial(norm_layer, affine=False, track_running_stats=False)

        import model.ResNet

        model_obj = model.ResNet.ResnetGenerator(norm_layer=norm_layer, **config_resnet)
    elif model_choice == 'ynet':
        config_ynet = config_param['model']['config_gan']['config_ynet']
        import model.YNet
        model_obj = model.YNet.YNet(**config_ynet)
    else:
        model_obj = None

    model_obj = model_obj.float()

    if gan_ind:
        state_dir = os.path.join(model_path, 'state_generator_temp_weights.pt')
    else:
        state_dir = os.path.join(model_path, 'temp_weights.pt')

    state_dict = torch.load(state_dir, map_location='cpu')
    model_obj.load_state_dict(state_dict)


"""
Load the model..
"""

# Recreate the modelling object
# A = decision_obj.decision_maker()
# A.load_weights()


import reconstruction.ReadCpx as read_cpx
import helper.array_transf as htransf
dir_path = '/home/bugger/Documents/data/7T/prostate/2020_05_27/V9_10168'
dir_path = '/home/bugger/Documents/data/7T/prostate/2020_06_17/ph_10930'
list_files = os.listdir(dir_path)
i_file = [x for x in list_files if 'v9_27052020_1556258_14_2_radwfsminlesssaturV4' in x and x.endswith('cpx')][0]
i_file = [x for x in list_files if 'v9_27052020_1559116_16_2_radlesssatur50V4' in x and x.endswith('cpx')][0]
i_file = [x for x in list_files if 'ph_17062020_1709280_12_3_radial10wfsmaxV4' in x and x.endswith('cpx')][0]
i_file = [x for x in list_files if 'ph_17062020_1709373_13_3_radial50wfsminV4' in x and x.endswith('cpx')][0]


for i in sorted(list_files):
    if i.endswith('.cpx'):
        print(i)

file_path = os.path.join(dir_path, i_file)
print_output = True
# # #

A, A_list = read_cpx.read_cpx_img(file_path, sel_loc=[0])
hplotc.SlidingPlot(A)
# THis is the right order... for ph_10930 study
A_stack = np.stack([A[-8:][7], A[-8:][6], A[-8:][4], A[-8:][5],
                    A[-8:][3], A[-8:][0], A[-8:][1], A[-8:][2]], axis=0)
# RIght order for V9_10168 study
# A_stack = np.stack([A[-8:][0], A[-8:][1], A[-8:][3], A[-8:][2], *A[-8:][4:]], axis=0)
A_stack = A_stack / np.max(np.abs(A_stack))
hplotc.SlidingPlot(A[-8:])
hplotc.SlidingPlot(A_stack)

im_y, im_x = A_stack.shape[-2:]
A_stacked = htransf.to_stacked(A_stack, cpx_type='cartesian', stack_ax=0)
A_stacked = A_stacked.T.reshape((im_x, im_y, -1)).T
# plt.imshow(np.swapaxes(A_stacked, -1, -2)[0, ::-1, ::-1])

# A_tensor = torch.from_numpy(np.swapaxes(A_stacked, -1, -2)[:, ::-1, ::-1][np.newaxis].copy()).float()
A_tensor = torch.as_tensor(A_stacked[np.newaxis]).float()

res = model_obj(A_tensor)
A_pred_abs = res[0][0].detach().numpy()
hplotf.plot_3d_list([np.squeeze(A_stack), np.squeeze(A_stack).sum(axis=0), A_pred_abs], augm='np.abs')
plt.imshow(A_pred_abs)

# # #

A.test_loader.dataset.file_list
x0, y0, y_pred0, plot_augmentation = A.get_image_prediction(0)
for i_augm in ['np.real', 'np.imag', 'np.abs', 'np.angle']:
    hplotf.plot_3d_list([x0, y0, y_pred0], augm=i_augm)

import helper.array_transf as harray
c_cpx = harray.to_complex(np.moveaxis(y_pred0[0], 0, -1))
y_cpx = harray.to_complex(np.moveaxis(y0[0], 0, -1))
x_cpx = harray.to_complex(np.moveaxis(x0[0], 0, -1))
for i_augm in ['np.real', 'np.imag', 'np.abs', 'np.angle']:
    hplotf.plot_3d_list([x_cpx, y_cpx, c_cpx], augm=i_augm)

# Get more stuff out of there....
hplotf.plot_3d_list([x0, y0, y_pred0], augm='np. abs')
plot_img = c_cpx
temp_plot = [np.real(plot_img), np.imag(plot_img), np.angle(plot_img), np.abs(plot_img)]
plot_img = y_cpx
temp_plot_target = [np.real(plot_img), np.imag(plot_img), np.angle(plot_img), np.abs(plot_img)]
temp_plot_diff = [np.abs(x-y) for x, y in zip(temp_plot, temp_plot_target)]

hplotf.plot_3d_list([np.stack(temp_plot, axis=0), np.stack(temp_plot_target, axis=0), np.stack(temp_plot_diff, axis=0)],
                    subtitle=[list(['prediction - real', 'prediction - imag', 'prediction - angle', 'prediction - abs']),
                              list(['target - real', 'target - imag', 'target - angle', 'target - angle']),
                              list(['difference - real', 'difference - imag', 'difference - angle', 'difference - angle'])],
                    ax_off=True, cbar=True)


# Get intermediate layer results..
import helper_torch.misc as htmisc
res = htmisc.get_all_children(A.generator)

import helper.array_transf as harray

x_0 = torch.as_tensor(x0).float()
result_interm = []
with torch.no_grad():
    for i_layer in res:
        x_0 = i_layer(x_0)
        result_interm.append(x_0)

for i in result_interm:
    print(i.shape)

n_layers = len(result_interm)
for i in np.linspace(0, len(result_interm), 10):
# for i in range(n_layers - 10, n_layers):
    index = int(i)
    hplotf.plot_3d_list(result_interm[index], title=str(res[index]))

res_a = res[48](torch.as_tensor(a)[np.newaxis])
conv_layer = torch.nn.Conv2d(in_channels=2, out_channels=256, kernel_size=3)
conv_layer.weight = torch.nn.Parameter(torch.as_tensor(a_np_weight[:, 0:2, :, :]))

res_a = conv_layer(torch.as_tensor(a)[np.newaxis])
hplotf.plot_3d_list(res_a[:, 0:100].detach().numpy())

index = 48
a_weights = list(res[48].parameters())
hplotf.plot_3d_list(result_interm[index], title=str(res[index]))
result_interm[48].shape
a_np_weight = a_weights[0].detach().numpy()
hplotf.plot_3d_list(a_np_weight[255:256, :])