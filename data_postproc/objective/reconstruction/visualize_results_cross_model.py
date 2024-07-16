import getpass

if getpass.getuser() == 'sharreve':
    import matplotlib
    matplotlib.use('Agg')

from objective_helper.reconstruction import prepare_metric_dataset, convert_to_sos
import os
import helper.plot_class as hplotc
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import helper.misc as hmisc
from objective_configuration.reconstruction import DRESULT, ACCELERATION_LIST, TYPE_NAMES, MODEL_NAMES, \
    MODEL_COLOR_DICT, MODEL_NAME_DICT, METRIC_NAMES, FONTSIZE_XTICKS, FONTSIZE_YTICKS, PERCENTAGE_LIST, DRESULT_INFERENCE, DINFERENCE, DDATA, DRECON
import objective_helper.reconstruction as hrecon


"""
We want to visualize the following...

input   model_1 model_2 .. model_n target

For 100p models


"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # It is either acc or inference
    parser.add_argument('-type', '-t', type=str, help='Model type name')
    parser.add_argument('-percentage', '-p', type=str, help='Model type name', default=100, required=False)
    parser.add_argument('-acc', '-a', type=str, help='5, 10', required=False)
    parser.add_argument('-number', '-n', type=str, help='Number of files to store', default=None)
    parser.add_argument('--inference', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--scaling', default=False, action='store_true')
    parser.add_argument('--paper', default=False, action='store_true')

    # Parses the input
    p_args = parser.parse_args()
    debug = p_args.debug
    N = int(p_args.number) if p_args.number is not None else None
    inference_bool = p_args.inference
    acc = p_args.acc
    sel_type = p_args.type
    sel_percentage = p_args.percentage
    scaling_bool = p_args.scaling
    paper_bool = p_args.paper

    if inference_bool:
        pred_folder = DRESULT_INFERENCE
        ddata = ddata_input = os.path.join(DINFERENCE, 'input')
        ddata_target = os.path.join(DINFERENCE, 'target')
    else:
        pred_folder = DRESULT
        ddata = ddata_input = ddata_target = os.path.join(DDATA, 'mixed', 'test', 'input')

    file_list = [x for x in os.listdir(ddata) if x.endswith('h5')]
    # First fix inference only
    subtitle_list = ['input'] + [MODEL_NAME_DICT[x] for x in MODEL_NAMES[:2]] + ['target', ''] + [MODEL_NAME_DICT[x] for x in MODEL_NAMES[2:]]+ ['']
    for jj, i_file in enumerate(file_list[:N]):
        base_name = hmisc.get_base_name(i_file)
        print(i_file)
        # Find input and target too...
        input_array = hmisc.load_array(os.path.join(ddata_input, i_file), data_key='kspace', sel_slice='mid')
        if inference_bool:
            input_sos = convert_to_sos(input_array)
            acc_str = 'inference' if inference_bool else 'retro'
        else:
            # Undersample
            acc_str = acc
            n_points = input_array.shape[0]
            if acc == '5':
                us_radial_traj = hrecon.undersample_trajectory(img_size=input_array.shape[:2], n_points=n_points, p_undersample=100 // 5)
            else:
                us_radial_traj = hrecon.undersample_trajectory(img_size=input_array.shape[:2], n_points=n_points, p_undersample=100 // 10)

            # input_sos = convert_to_sos(input_array)
            input_cpx = np.fft.ifftn(np.fft.fftshift(input_array[..., ::2] + 1j * input_array[..., 1::2]), axes=(-3, -2))
            input_cpx = np.moveaxis(input_cpx, -1, 0)
            input_array = hrecon.undersample_img(input_cpx, traj=us_radial_traj)
            input_sos = np.sqrt(np.sum(np.abs(input_array) ** 2, axis=0))

        target_array = hmisc.load_array(os.path.join(ddata_target, i_file), data_key='kspace', sel_slice='mid')
        target_sos = convert_to_sos(target_array)

        plot_array = [input_sos]
        for i_model in MODEL_NAMES:
            if inference_bool:
                sub_str = f'train_mixed/undersampled'
            else:
                sub_str = f'train_mixed/mixed/{acc}x'

            img_folder = f'{i_model}{sel_type}/{str(sel_percentage)}p/{sub_str}/{i_file}'
            img_path = os.path.join(pred_folder, img_folder)
            if os.path.isfile(img_path):
                temp_array = hmisc.load_array(img_path, data_key='reconstruction', sel_slice='mid')
            else:
                print('Not found ', img_path)
                temp_array = np.zeros(input_sos.shape)

            # Rotate the array
            plot_array.append(temp_array)

        # This is to hopefully make the arrya look like this
        # input Model1 Model2 target
        # none  Model3 Model4 none
        plot_array.insert(3, target_sos)
        plot_array.insert(4, np.zeros(target_sos.shape))
        plot_array.append(np.zeros(target_sos.shape))
        plot_array.append(target_sos)
        # Rotate all content
        plot_array = [x[::-1, ::-1] for x in plot_array]
        # Plot array is now ready...
        # fig_obj = hplotc.ListPlot(plot_array, ax_off=True, col_row=(len(plot_array), 1), wspace=0)
        # fig_obj.figure.savefig(os.path.join(ddest, base_name + '.png'))

        plot_folder = pred_folder
        if paper_bool:
            plot_folder = os.path.join(DRECON, 'Figure 2a - example of models')

        temp_height = 3 * 0.05
        fig_obj = hplotc.ListPlot(np.array(plot_array), col_row=(4, 2), ax_off=True, figsize=(15, 6), aspect='auto', hspace=temp_height, wspace=0,
                                     ddest=plot_folder, subtitle_list=subtitle_list, proper_scaling=scaling_bool,
                                     only_str=True)

        for jsubt, i_subtitle in enumerate(subtitle_list):
            import helper.plot_fun as hplotf
            if i_subtitle:
                hplotf.add_text_box(fig_obj.figure, jsubt, str(i_subtitle), height_rect=0.05,
                                    linewidth=1, position='top', fontsize=FONTSIZE_XTICKS)

        str_appendix = f'pred_cross_model_type_{sel_type}_acc_{acc}_perc_{sel_percentage}_' + base_name
        if paper_bool:
            str_appendix = f'{acc_str}_{sel_type}_image_{jj}'

        fig_obj.savefig(os.path.join(plot_folder, str_appendix), home=False)

