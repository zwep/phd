import torch
import sys
import shutil
import sys
import os
import json
import getpass
import subprocess
import argparse

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

print('Adding to path: ', project_path)
sys.path.append(project_path)

import helper.misc as hmisc
import helper.plot_fun as hplotf
import helper.nvidia_parser as hnvidia
import helper.model_setting as hmodel_set

import objective.inhomog_removal.executor_inhomog_removal as executor  # ==>

"""
Lets see if we can create a file that executes batch scripts..
"""

parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str)
parser.add_argument('-debug', type=str)

# Parses the input
p_args = parser.parse_args()
debug = p_args.debug
if debug:
    print('DEBUG MODE IS ACTIVATED')
    debug_ind = True
else:
    debug_ind = None

path_config_files = p_args.path
list_config_files = os.listdir(path_config_files)

base_path = os.path.dirname(path_config_files)
base_name = os.path.basename(path_config_files)

for i_config_file in list_config_files:
    print('Processing config file ', i_config_file)
    file_name, _ = os.path.splitext(i_config_file)
    config_file = os.path.join(path_config_files, i_config_file)
    print('Full path of config file ', config_file)

    template_path = os.path.join(base_path, base_name + '_' + file_name)
    if not os.path.isdir(template_path):
        print('Creating directory ', template_path)
        os.mkdir(template_path)

    # Copy the config file to its desination
    print('Copy ', config_file)
    print('to ', template_path)
    shutil.copyfile(config_file, os.path.join(template_path, 'config_run.json'))

    # Reads the json
    with open(config_file, 'r') as f:
        text_obj = f.read()
        config_model = json.loads(text_obj)

    config_model['dir']['dtemplate'] = template_path
    if local_system:
        config_model['dir']['ddata'] = '/home/bugger/Documents/data/semireal/prostate_simulation_t1t2_rxtx'
    #
    # hmisc.print_dict(config_model)
    mult_dict = hmodel_set.create_mult_dict(config_model, **config_model)
    print('Model path is ', config_model['dir']['doutput'])
    model_path_list = hmodel_set.create_config_dir(config_model['dir']['doutput'], mult_dict, debug=True)
    #
    # """
    # Set session
    # """
    #
    index_gpu, p_gpu = hnvidia.get_free_gpu_id(claim_memory=config_model['gpu_frac'])
    if index_gpu is not None:
        print('Status GPU')
        nvidia_cmd = ["nvidia-smi", "-q", "-d", "MEMORY", "-i"]
        cmd = nvidia_cmd + [str(index_gpu)]
        output = subprocess.check_output(cmd).decode("utf-8")
        print(output)

    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #\n\n')

    print('Starting')
    # Take the one with the least amount of slices..
    print('Amount of configs created ', len(model_path_list))
    for full_model_path in model_path_list:
        print('Chosen config: ', full_model_path)

        # We now give an index GPU such that ALL the models will put their information on ONE gpu.
        # This should be the new way.. to distinguish between GANs and non GANs
        decision_obj = executor.DecisionMaker(model_path=full_model_path, debug=debug_ind, index_gpu=index_gpu)  # ==>>
        modelrun_obj = decision_obj.decision_maker()
        modelrun_obj.save_model(plot_name='initial')
        hplotf.close_all()
        history_dict = modelrun_obj.train_model()
        test_running_loss = modelrun_obj.test_model()
        fig_handle_img = modelrun_obj.save_model(plot_name='final')
        hplotf.close_all()
        fig_handle_loss = modelrun_obj.save_history_graph()
        fig_handle_grad = modelrun_obj.save_weight_history()

        hplotf.close_all()

        del modelrun_obj  # Clear some cache stuff
        # Removed this to see if things will go fine now..
        # torch.cuda.empty_cache()  # This occupies memory on GPU 0... which is annoying.
