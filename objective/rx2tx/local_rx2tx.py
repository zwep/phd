# encoding: utf-8

"""
Loading packages
"""

import numpy as np
import os
import sys
import getpass
import json
import importlib
import torch
import gc

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
else:
    project_path = "/home/seb/code/pytorch_in_mri"

model_path = os.path.join(project_path, 'objective/rx2tx/configuration')
print('Adding to path: ', project_path)
sys.path.append(project_path)

print('WE ARE IN MANUAL MODE, some functionalities may not work as intended')

"""
Load the project dependent packages
"""

import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.misc as hmisc
import helper.model_setting as hmodel_set
import helper.nvidia_parser as hnvidia
import torch.utils.data
import objective.rx2tx.executor_rx2tx as executor


"""
Load config file
"""

config_file_name = "rx2tx.json"  # ==>>
config_file = os.path.join(model_path, config_file_name)

with open(config_file, 'r') as f:
    text_obj = f.read()
    config_model = json.loads(text_obj)

if local_system:
    # With this we store the data in the project directory
    config_model['dir']['dtemplate'] = model_path
    config_model['dir']['doutput'] = "/home/bugger/Documents/model_run/test_results"
    # config_model['dir']['ddata'] = "/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels"  # ==>>
    config_model['dir']['ddata'] = '/home/bugger/Documents/data/simulation/prostate_mri_mrl'
else:
    config_model['dir']['doutput'] = "/data/seb/model_run/test_run"
    config_model['dir']['dtemplate'] = model_path
    # config_model['dir']['ddata'] = "/home/seb/data/b1shimsurv_all_channels"  # ==>>
    config_model['dir']['ddata'] = "/data/seb/flavio_npy" # ==>>

config_model['model']['n_epoch'] = 1
debug_ind = False

# Print the loaded config
hmisc.print_dict(config_model)

"""
Unpack config model
"""

mult_dict = hmodel_set.create_mult_dict(config_model, **config_model, debug=debug_ind)
model_path_list = hmodel_set.create_config_dir(config_model['dir']['doutput'], mult_dict, debug=debug_ind)

"""
Set session
"""

index_gpu, p_gpu = hnvidia.get_free_gpu_id(claim_memory=config_model['gpu_frac'])
device = torch.device("cuda:{}".format(str(index_gpu)) if torch.cuda.is_available() else "cpu")

# Take the one with the least amount of slices..
print('Amount of configs created ', len(model_path_list))
for full_model_path in model_path_list:
    print('Chosen config: ', full_model_path)
    # full_model_path = model_path_list[0]
    importlib.reload(executor)

    # This should be the new way.. to distinguish between GANs and non GANs
    decision_obj = executor.DecisionMaker(model_path=full_model_path, debug=True, index_gpu=index_gpu)  # ==>>
    modelrun_obj = decision_obj.decision_maker()

    x0, y0, y_pred0, plot_augmentation = modelrun_obj.get_image_prediction(sel_batch=0)
    cont = modelrun_obj.train_loader.dataset.__getitem__(0)
    cont['target'].shape
    modelrun_obj.train_loader.dataset.transform_type_target
    hplotf.plot_3d_list(y_pred0, augm='np.angle')

    print('Executor GAN shape of img pred ', x0.shape, y0.shape, y_pred0.shape)
    if x0.ndim > 2:
        # For now.. just assume that if we have a larger dimension.. then the amount of channels will always be in the first dim.
        n_chan = x0.shape[0]
    else:
        n_chan = -1
        print('Unkown output dimension of x0 ', x0.shape)

    doutput = modelrun_obj.config_param['dir']['doutput']
    plot_name = 'initial'
    # This should be a complex valued array...?
    try:
        plot_array = np.stack([x0, y0, y_pred0], axis=0)
    except ValueError:
        plot_array = [x0, y0, y_pred0]

    for i_augm in plot_augmentation:
        output_path = os.path.join(doutput, f"{plot_name}_{i_augm}.jpg")
        fig_handle = hplotf.plot_3d_list(plot_array, figsize=(15, 10), dpi=75, augm=i_augm,
                                         subtitle=[['input'] * n_chan, ['target'] * n_chan, ['pred'] * n_chan],
                                         title=i_augm)
        # fig_handle.savefig(output_path)

    del plot_array  # Frees up a lot of space...

    history_dict = modelrun_obj.train_model()
    test_running_loss = modelrun_obj.test_model()
    fig_handle_img = modelrun_obj.save_model()
    fig_handle_loss = modelrun_obj.save_history_graph()
    fig_handle_grad, fig_handle_param = modelrun_obj.save_weight_history()
    plt.close('all')


self = modelrun_obj
overal_grad = self.history_dict['overal_grad']

overal_grad_switch = hmisc.change_list_order(overal_grad)
n_layers = len(overal_grad_switch)
summary_grad = []
for i_layer in overal_grad_switch:
    temp_data = []
    for i_time in i_layer:
        t1 = np.min(i_time)
        t2 = np.mean(i_time)
        t3 = np.max(i_time)
        res = (t1, t2, t3)
        temp_data.append(res)

    summary_grad.append(temp_data)

fig, ax = plt.subplots(1, n_layers)
ax = ax.ravel()
for i in range(n_layers):
    temp_min, temp_mean, temp_max = zip(*summary_grad[i])
    ax[i].plot(temp_min, alpha=0.5)
    ax[i].plot(temp_mean, '-.')
    ax[i].plot(temp_max, alpha=0.5)
    ax[i].set_ylim(0, 0.1)
    ax[i].set_axis_off()


# MiB to byte
# 1048576
#
# from objective.recall_base import RecallBase
# recall_obj = RecallBase(config_run_file=config_model)
# recall_obj.write_test_result()
# recall_obj.write_figure()


