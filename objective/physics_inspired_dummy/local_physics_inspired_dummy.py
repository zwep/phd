# encoding: utf-8

"""
Running physics_inspired_dummy
"""

# Standard packages
import os
import sys
import json
import argparse
import getpass
import subprocess
# Self created code
import helper.misc as hmisc
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.nvidia_parser as hnvidia
import helper.model_setting as hmodel_set

import objective.physics_inspired_dummy.executor_physics_inspired_dummy as executor  # ==>
import objective.recall_base as recall_base
config_file_name = "physics_inspired_dummy.json"  # ==>>


def plot_two_complex_lines(line_1, line_2):
    fig, axes = plt.subplots(4)
    axes[0].plot(line_1.real)
    axes[0].plot(line_2.real)
    axes[0].set_title('real')
    axes[1].plot(line_1.imag)
    axes[1].plot(line_2.imag)
    axes[1].set_title('imag')
    axes[2].plot(np.abs(line_1))
    axes[2].plot(np.abs(line_2))
    axes[2].set_title('abs')
    axes[3].plot(np.angle(line_1))
    axes[3].plot(np.angle(line_2))
    axes[3].set_title('angle')
    return fig


if __name__ == '__main__':
    """
    Parse the arguments
    """
    config_file = '~/PycharmProjects/pytorch_in_mri/objective/physics_inspired_dummy/configuration/physics_inspired_dummy.json'
    config_file = os.path.expanduser(config_file)
    template_path = os.path.dirname(config_file)
    debug_ind = True

    # Reads the json
    with open(config_file, 'r') as f:
        text_obj = f.read()
        config_model = json.loads(text_obj)

    config_model['dir']['dtemplate'] = template_path
    # config_model['dir']['ddata'] = None
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
        decision_obj = executor.DecisionMakerPhysicInspiredDummy(model_path=full_model_path, debug=False, index_gpu=index_gpu)  # ==>>
        modelrun_obj = decision_obj.decision_maker()

        modelrun_obj.save_model(plot_name='initial')

        history_dict = modelrun_obj.train_model()

        # # #
        # Live coding of 1D example...
        import numpy as np
        import torch
        fig, axes = plt.subplots(2)
        for _ in range(5):
            input_dict = modelrun_obj.train_loader.dataset.__getitem__(_)
            input_tensor = input_dict['input']
            target_tensor = input_dict['target']
            input_np = input_tensor.numpy()
            input_tensor[:, -10:] = 0
            output_tensor = modelrun_obj.model_obj(input_tensor).detach()
            output_np = output_tensor.numpy()

            input_cpx = input_np[0] + 1j * input_np[1]
            output_cpx = output_np[0] + 1j * output_np[1]
            plot_two_complex_lines(input_cpx, output_cpx)


        modelrun_obj.loss_obj(output_tensor[None], input_dict['target'][None])
        output_xx = modelrun_obj.loss_obj.operator_xx(output_tensor[None])
        plt.plot(output_xx.numpy()[0][0])
        plt.plot(output[0])


        # # #
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
