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

model_path = os.path.join(project_path, 'objective/unet_validation/configuration')
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

import objective.unet_validation.executor_unet_validation as executor


"""
Load config file
"""

config_file_name = "unet_validation.json"  # ==>>
config_file = os.path.join(model_path, config_file_name)

with open(config_file, 'r') as f:
    text_obj = f.read()
    config_model = json.loads(text_obj)

if local_system:
    # With this we store the data in the project directory
    config_model['dir']['dtemplate'] = model_path
    config_model['dir']['doutput'] = "/home/bugger/Documents/model_run/test_results"
    config_model['dir']['ddata'] = "/home/bugger/Documents/data/grand_challenge/data"
else:
    config_model['dir']['doutput'] = "/data/seb/model_run/test_run"
    config_model['dir']['dtemplate'] = model_path
    config_model['dir']['ddata'] = "/home/seb/data/grand_challenge/data"  # ==>>

config_model['model']['n_epoch'] = 2
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
    importlib.reload(executor)
    full_model_path = model_path_list[-1]

    modelrun_obj = executor.ExecutorUnetValidation(model_path=full_model_path, debug=True, index_gpu=index_gpu)  # ==>>

    a, b, c = modelrun_obj.get_image_prediction()
    a.shape
    b.shape
    c.shape
    import helper.plot_class as hplotc

    hplotc.SlidingPlot(np.moveaxis(b[0][0], -1, 0))
    hplotc.SlidingPlot(np.moveaxis(c[0][0], -1, 0))
    modelrun_obj.loss_obj(torch.as_tensor(c), torch.as_tensor(b))
    import helper_torch.loss as hloss
    importlib.reload(hloss)
    loss_dice = hloss.DiceLoss()
    modelrun_obj.model_obj.eval()  # IMPORTANT
    with torch.no_grad():  # IMPORTANT
        torch_pred = modelrun_obj.model_obj(modelrun_obj.x_val)

    loss_dice(torch_pred, modelrun_obj.y_val)
    modelrun_obj.save_model()

    model_parameters = filter(lambda p: p.requires_grad, modelrun_obj.model_obj.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print('Chosen model: ', modelrun_obj.config_param['model']['model_choice'])
    print('Number of parameters', params)

    # with torch.no_grad():
    history_dict = modelrun_obj.train_model()

    test_running_loss = modelrun_obj.test_model()
    fig_handle_img = modelrun_obj.save_model()
    fig_handle_loss = modelrun_obj.save_history_graph()
    plt.close('all')


from objective.recall_base import RecallBase
recall_obj = RecallBase(config_run_file=config_model)
recall_obj.write_test_result()
recall_obj.write_figure()


