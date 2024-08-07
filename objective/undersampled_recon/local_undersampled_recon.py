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

model_path = os.path.join(project_path, 'objective/undersampled_recon/configuration')
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
import objective.undersampled_recon.executor_undersampled_recon as executor  # ==>>


"""
Load config file
"""

config_file_name = "undersampled_recon.json"  # ==>>
config_file = os.path.join(model_path, config_file_name)

with open(config_file, 'r') as f:
    text_obj = f.read()
    config_model = json.loads(text_obj)

if local_system:
    # With this we store the data in the project directory
    config_model['dir']['dtemplate'] = model_path
    config_model['dir']['doutput'] = "/home/bugger/Documents/model_run/test_results"
    # config_model['dir']['ddata'] = "/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels"  # ==>>
    # config_model['dir']['ddata'] = '/home/bugger/Documents/data/simulation/flavio_npy'
    # config_model['dir']['ddata'] = '/home/bugger/Documents/data/semireal/prostate_simulation_h5'
    # config_model['dir']['ddata'] = '/home/bugger/Documents/data/semireal/prostate_simulation_rxtx'
    # config_model['dir']['ddata'] = '/home/bugger/Documents/data/fastmri'
    config_model['dir']['ddata'] = '/media/bugger/MyBook/data/semireal/prostate_simulation_h5'
    config_model['dir']['ddata'] = '/media/bugger/MyBook/data/7T_data/radial_dataset_4ch'
else:
    config_model['dir']['dtemplate'] = model_path
    config_model['dir']['doutput'] = "/data/seb/model_run/test_run"
    config_model['dir']['ddata'] = "/data/seb/semireal/prostate_simulation_h5"  # ==>>

config_model['model']['n_epoch'] = 1
config_model['data']['number_of_examples'] = 1
debug_ind = True

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
    # full_model_path = model_path_list[0]

    # This should be the new way.. to distinguish between GANs and non GANs
    decision_obj = executor.DecisionMakerRecon(model_path=full_model_path, debug=True, index_gpu=index_gpu, inference=True)  # ==>>
    modelrun_obj = decision_obj.decision_maker()
    modelrun_obj.test_model()

    history_dict = modelrun_obj.train_model()
    a, b = modelrun_obj.test_loader.dataset.__getitem__(0)
    with torch.no_grad():
        res = modelrun_obj.generator(a[np.newaxis])

    with torch.no_grad():
        derp = modelrun_obj.discriminator(res)

    plt.imshow((res[0][0].detach().numpy()))
    plt.imshow(np.log(derp[0][0].detach().numpy()))

    x0, y0, y_pred0, plot_augmentation = modelrun_obj.get_image_prediction()
    if x0.ndim == 3:
        # Dit hoor ik niet meer te hebben toch...?
        # In some case we have data that is different..
        n_chan, _, _ = x0.shape
    else:
        print('Unkown output dimension of x0 ', x0.shape)

    # This should be a complex valued array...?
    plot_array = np.stack([x0, y0, y_pred0], axis=0)
    for i_augm in plot_augmentation:
        fig_handle = hplotf.plot_3d_list(plot_array, figsize=(15, 10), dpi=75, augm=i_augm,
                                         subtitle=[['input'] * n_chan, ['target'] * n_chan, ['pred'] * n_chan], title=i_augm)
        fig_handle.savefig(os.path.join(full_model_path, f"initial_results_{i_augm}.jpg"))

    del plot_array  # Frees up a lot of space...

    history_dict = modelrun_obj.train_model()
    test_running_loss = modelrun_obj.test_model()
    fig_handle_img = modelrun_obj.save_model()
    fig_handle_loss = modelrun_obj.save_history_graph()
    plt.close('all')

# MiB to byte
# 1048576
#
# from objective.recall_base import RecallBase
# recall_obj = RecallBase(config_run_file=config_model)
# recall_obj.write_test_result()
# recall_obj.write_figure()


