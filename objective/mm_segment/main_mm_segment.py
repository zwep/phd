# encoding: utf-8

"""
Running mm_segment
"""

# Standard packages
import os
import sys
import json
import argparse
import getpass
import subprocess

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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Self created code
import helper.misc as hmisc
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.nvidia_parser as hnvidia
import helper.model_setting as hmodel_set

import objective.mm_segment.executor_mm_segment as executor  # ==>
import objective.recall_base as recall_base
config_file_name = "mm_segment.json"  # ==>>

if __name__ == '__main__':
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str)
    parser.add_argument('-debug', type=str)

    # Parses the input
    p_args = parser.parse_args()
    config_file = p_args.config
    debug = p_args.debug
    if debug:
        print('DEBUG MODE IS ACTIVATED')
        debug_ind = True
    else:
        debug_ind = None

    template_path = os.path.dirname(config_file)

    # Reads the json
    with open(config_file, 'r') as f:
        text_obj = f.read()
        config_model = json.loads(text_obj)

    config_model['dir']['dtemplate'] = template_path
    if local_system:
        config_model['dir']['ddata'] = "/media/bugger/MyBook/data/m&m/MnM_dataset"
        config_model['dir']['doutput'] = "/home/bugger/Documents/model_run/test_run"  # ==>>

    # Print the loaded config
    hmisc.print_dict(config_model)

    """
    Unpack config model
    """
    mult_dict = hmodel_set.create_mult_dict(config_model, **config_model)
    model_path_list = hmodel_set.create_config_dir(config_model['dir']['doutput'], mult_dict)

    """
    Set session
    """

    index_gpu, p_gpu = hnvidia.get_free_gpu_id(claim_memory=config_model['gpu_frac'])
    if index_gpu is not None:
        print('Status GPU')
        nvidia_cmd = ["nvidia-smi", "-q", "-d", "MEMORY", "-i"]
        cmd = nvidia_cmd + [str(index_gpu)]
        output = subprocess.check_output(cmd).decode("utf-8")
        print(output)

    print('Starting')
    # Take the one with the least amount of slices..
    print('Amount of configs created ', len(model_path_list))
    for full_model_path in model_path_list:
        print('Chosen config: ', full_model_path)

        # We now give an index GPU such that ALL the models will put their information on ONE gpu.
        # This should be the new way.. to distinguish between GANs and non GANs
        decision_obj = executor.DecisionMakerMMSegment(model_path=full_model_path, debug=debug_ind, index_gpu=index_gpu)  # ==>>
        modelrun_obj = decision_obj.decision_maker()

        modelrun_obj.save_model(plot_name='initial')

        history_dict = modelrun_obj.train_model()
        test_running_loss = modelrun_obj.test_model()
        fig_handle_img = modelrun_obj.save_model(plot_name='prediction')
        fig_handle_loss = modelrun_obj.save_history_graph()
        fig_handle_grad = modelrun_obj.save_weight_history()

        plt.close('all')

        del modelrun_obj  # Clear some cache stuff
        # torch.cuda.empty_cache()  # This occupies memory on GPU 0... which is annoying.

    recall_obj = recall_base.RecallBase(config_run_file=config_model)
    recall_obj.write_test_result()
    recall_obj.write_figure()
