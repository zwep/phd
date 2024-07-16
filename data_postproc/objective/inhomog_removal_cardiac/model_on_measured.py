"""
Test how well we can remove bias field stuff on 4Ch things..
"""
import helper.array_transf

"""
Here we are going to collect all the model outputs that give the undistrubed/restored as result immediately

"""

import matplotlib.pyplot as plt
import helper.plot_class as hplotc
import numpy as np
import torch
import helper.misc as hmisc
import helper.array_transf as harray
import objective.inhomog_removal.executor_inhomog_removal as executor
import os

"""
Stuff
"""

# Choose with which model we are going to generate the results
model_path_dir = '/home/bugger/Documents/model_run/inhom_removal_4ch' # ==>
ddata = '/media/bugger/MyBook/data/7T_data/cartesian_radial_dataset_4ch/train/target'
file_list = [os.path.join(ddata, x) for x in os.listdir(ddata)]
model_name = os.path.basename(model_path_dir)

config_param = hmisc.convert_remote2local_dict(model_path_dir, path_prefix='/media/bugger/MyBook/data/semireal')
# Otherwise squeeze will not work properly..
config_param['data']['batch_size'] = 1

"""
Load the model
"""

decision_obj = executor.DecisionMaker(config_file=config_param, debug=False,
                                      load_model_only=True, inference=True, device='cpu')  # ==>>
modelrun_obj = decision_obj.decision_maker()
modelrun_obj.load_weights()
target_type = modelrun_obj.config_param['data']['target_type']

if modelrun_obj.model_obj:
    modelrun_obj.model_obj.eval()
else:
    modelrun_obj.generator.eval()

image_result = []
metric_result = []

model_debug_counter = 0
debug_plot_data = 0
i = 1
for i_file in file_list:
    result_list = []
    input_array_card = np.load(i_file)
    # We have multiple cardiac phases....
    for input_cpx in input_array_card:
        x_input = input_cpx
        uncorrected_image = np.abs(input_cpx)
        n_y, n_x = x_input.shape[-2:]
        x_inputed = harray.to_stacked(x_input, cpx_type='cartesian', stack_ax=0)
        x_inputed = x_inputed.T.reshape((n_x, n_y, -1)).T
        A_tensor = torch.as_tensor(x_inputed[np.newaxis]).float()

        with torch.no_grad():
            if modelrun_obj.config_param['model']['model_choice'] == 'gan':
                output = modelrun_obj.generator(A_tensor)
            elif modelrun_obj.config_param['model']['model_choice'] == 'cyclegan':
                output = modelrun_obj.netG_A2B(A_tensor)
            else:
                output = modelrun_obj.model_obj(A_tensor)


        if (target_type == 'biasfield') or (target_type == 'expansion'):
            biasfield = output
            biasfield = harray.smooth_image(biasfield, n_kernel=16)
            corrected_image = uncorrected_image / biasfield
            corrected_image = helper.array_transf.correct_inf_nan(corrected_image)

            cutoff_perc = 5
            noise_mask = (biasfield / np.max(biasfield, axis=(-2, -1)) * 100) < cutoff_perc
            corrected_image[np.abs(corrected_image) > 1] = 1
            corrected_image[noise_mask] = 0
            hplotc.ListPlot([[biasfield, corrected_image, uncorrected_image]], vmin=(0, 1), ax_off=True)
        else:
            corrected_image = output
            biasfield = uncorrected_image / corrected_image
            biasfield = helper.array_transf.correct_inf_nan(biasfield)

            biasfield_smooth = harray.smooth_image(biasfield, n_kernel=32)
            corrected_image = uncorrected_image / biasfield_smooth
            corrected_image = helper.array_transf.correct_inf_nan(corrected_image)
            corrected_image = harray.scale_minmax(corrected_image)

        temp_dict = {'corrected': corrected_image,
                     'uncorrected': uncorrected_image,
                     'biasfield': biasfield}

        result_list.append(temp_dict)

    result_corrected = [x['corrected'] for x in result_list]
    result_corrected = [harray.scale_minmax(x) for x in result_corrected]
    result_uncorrected = [x['uncorrected'] for x in result_list]
    result_bias_field = [x['biasfield'] for x in result_list]