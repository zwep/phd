
"""
Test several measures..
"""

import torch
import matplotlib.pyplot as plt
import objective.inhomog_removal.executor_inhomog_removal as executor

import data_generator.InhomogRemoval as data_gen

import numpy as np
import helper.misc as hmisc

from helper.metric import get_metrics_input, get_metrics_target


def scale_with_image(x, x_ref):
    x_max_min = x.max() - x.min()
    x_ref_max_min = x_ref.max() - x_ref.min()
    x_scaled = (x - x.min()) / x_max_min * x_ref_max_min + x_ref.min()
    return x_scaled


"""
Set dir
"""

prefix_path_prostate = '/home/bugger/Documents/data/semireal'
prefix_path_cardiac = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx'

model_path = '/home/bugger/Documents/model_run/inhom_removal_undistorted/resnet'
prediction_mode = 'prostate'

if prediction_mode == 'prostate':
    prefix_path = prefix_path_prostate
elif prediction_mode == 'p2ch' or prediction_mode == '2ch':
    prefix_path = prefix_path_cardiac
elif prediction_mode == 'p4ch' or prediction_mode == '4ch':
    prefix_path = prefix_path_cardiac
else:
    model_path = ''
    prefix_path = ''

"""
Get the config param
"""

config_param = hmisc.convert_remote2local_dict(model_path, path_prefix=prefix_path)
# Otherwise squeeze will not work properly..
config_param['data']['batch_size'] = 1
config_param['data']['sel_slice'] = None
config_param['data']['target_type'] = 'rho'
config_param['data']['b1p_scaling'] = True
config_param['data']['alternative_input'] = None
"""
Load the model (ONLY the model)
"""

decision_obj = executor.DecisionMaker(config_file=config_param, debug=True, inference=True)  # ==>>
modelrun_obj = decision_obj.decision_maker()
modelrun_obj.load_weights()

"""
Define data generator
"""

# Inhomogeneity removal the old-skool way on Gradient Echo data.
dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation_rxtx'
gen = data_gen.DataGeneratorInhomogRemoval(ddata=dir_data, dataset_type='test', complex_type='cartesian',
                                          input_shape=(1, 256, 256),
                                          use_tx_shim=True,
                                          b1m_scaling=False,
                                          b1p_scaling=True,
                                          debug=False,
                                          masked=True,
                                          lower_prob=0.0,
                                          target_type='rho')


# for container in gen:
test_loss = {}
for container in modelrun_obj.test_loader:
    input_cpx = container['input'][0].numpy()[::2] + 1j * container['input'][0].numpy()[1::2]
    input_sum = input_cpx.sum(axis=0)
    input_abs = np.abs(input_sum)
    target = container['target'].numpy()[0][0]

    # Baseline metrics...
    result_basic = get_metrics_target(input_abs, target)

    with torch.no_grad():
        res = modelrun_obj.model_obj(container['input'])

    output = res.numpy()[0][0]
    output_scaled_input = scale_with_image(output, input_abs)
    output_scaled_target = scale_with_image(output, target)
    result_model_target = get_metrics_target(output_scaled_target, target)
    result_model_input = get_metrics_input(output_scaled_input, input_abs)

    # Histogram Equalization
    import cv2
    output_he = cv2.equalizeHist((input_abs*255).astype(np.uint8))
    output_he_scaled_input = scale_with_image(output_he, input_abs)
    output_he_scaled_target = scale_with_image(output_he, target)
    result_he_target = get_metrics_target(output_he_scaled_target, target)
    result_he_input = get_metrics_input(output_he_scaled_input, input_abs)

    # Homomorfic filtering.... (using Gaussian kernel..)
    import biasfield_algorithms.HF as model_hf
    output_hf = model_hf.get_hf(input_abs, n_kernel=32)
    output_hf_scaled_target = scale_with_image(output_hf, target)
    output_hf_scaled_input = scale_with_image(output_hf, input_abs)
    result_hf_target = get_metrics_target(output_hf_scaled_target, target)
    result_hf_input = get_metrics_input(output_hf_scaled_input, input_abs)

    # Homomorfic unsharp masking.... (Using Butterworths)
    import biasfield_algorithms.HUM as model_holom
    output_holom = model_holom.get_holomorfic(input_abs)
    output_holom_scaled_target = scale_with_image(output_holom, target)
    output_holom_scaled_input = scale_with_image(output_holom, input_abs)
    result_holom_target = get_metrics_target(output_holom_scaled_target, target)
    result_holom_input = get_metrics_input(output_holom_scaled_input, input_abs)

    # PABIC fitting
    import biasfield_algorithms.PABIC as model_pabic
    output_pabic = model_pabic.get_lst_sqr(input_abs)
    output_pabic[np.isnan(output_pabic)] = 0
    output_pabic_scaled_target = scale_with_image(output_pabic, target)
    output_pabic_scaled_input = scale_with_image(output_pabic, input_abs)
    result_lm_target = get_metrics_target(output_pabic_scaled_target, target)
    result_lm_input = get_metrics_target(output_pabic_scaled_input, input_abs)

    # N4ITK standard method...
    # SImple wrapper for n4itk...
    import biasfield_algorithms.N4ITK as model_n4itk
    output_n4itk = model_n4itk.get_n4itk(input_abs)

    output_n4itk_scaled_target = scale_with_image(output_n4itk, target)
    output_n4itk_scaled_input = scale_with_image(output_n4itk, input_abs)
    result_n4itk_target = get_metrics_target(output_n4itk_scaled_target, target)
    result_n4itk_input = get_metrics_input(output_n4itk_scaled_input, input_abs)



    # OVerview of all the metrics.....
    list_result = [result_basic, result_lm, result_holom, result_model, result_n4itk]
    list_labels = ['basic', 'lm', 'holom', 'model', 'n4itk']
    list_result_values = [list(x.values()) for x in list_result]
    list_result_values = hmisc.change_list_order(list_result_values)
    list_result_dict = dict(zip(list_result[0].keys(), list_result_values))

    n_result = len(list_result_values)
    width = 0.1
    x_offset = np.arange(0, n_result*width, width) - np.ceil(n_result / 2) * width / 2

    for k, v in list_result_dict.items():
        fig, ax = plt.subplots()
        for i, i_result in enumerate(v):
            ax.bar(x_offset[i], i_result, width, label='Men')
            ax.set_xticks(x_offset)
            ax.set_xticklabels(list_labels)

        fig.suptitle(k)

