import getpass

if getpass.getuser() == 'sharreve':
    import matplotlib
    matplotlib.use('Agg')

from objective_helper.reconstruction import prepare_metric_dataset, setup_metric_figure
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from objective_configuration.reconstruction import DRESULT, ACCELERATION_LIST, TYPE_NAMES, MODEL_NAMES, \
    MODEL_COLOR_DICT, MODEL_COLOR_DICT_DARK, METRIC_NAMES, FONTSIZE_XTICKS, FONTSIZE_YTICKS, PERCENTAGE_LIST, DRESULT_INFERENCE, \
    MODEL_NAME_DICT, PRETR_SYNTH_ACQ_APPENDIX, FONTSIZE_YLABEL, DRECON, METRIC_NAME_DICT, FONTSIZE_LEGEND

"""

Compare between models, selecting only the 100p moidels

"""


def plot_bar(ax, x_pos, y_mean, y_std, model_name):
    model_plot_name = MODEL_NAME_DICT[model_name]
    color = MODEL_COLOR_DICT[model_name]
    color_dark = MODEL_COLOR_DICT_DARK[model_name]
    z = ax.bar(x_pos, y_mean, label=model_plot_name, zorder=5, color=color)
    # # Get the original color
    # original_color = z[0].get_facecolor()
    # # Create a darker color by multiplying the original RGB values by the factor
    # darker_color = [c * darkness_factor for c in original_color[:-1]] + [1]
    _ = ax.errorbar(x_pos, y_mean, y_std, lw=2, capsize=5, capthick=2, color=color_dark)
    return ax


def get_dict_of_datasets(selected_path, sel_type, anatomy_str, percentage_int):
    # Get the metrics dicts of all models belonging to a selected type (sel_type)
    assert sel_type in TYPE_NAMES
    dict_of_datasets = {}
    #
    for i_model in MODEL_NAMES:
        model_type = f'{i_model}{sel_type}'
        pred_folder = os.path.join(selected_path, model_type)
        djson = os.path.join(pred_folder, 'metric.json')
        prepped_dataset = prepare_metric_dataset(djson)
        # Make sure we only take the 'mixed' dataset
        if prepped_dataset is not None:
            sel_dataframe = prepped_dataset[prepped_dataset['anatomy'] == anatomy_str]
            sel_dataframe = sel_dataframe[sel_dataframe['percentage'] == percentage_int]
        else:
            sel_dataframe = prepped_dataset
        dict_of_datasets[model_type] = sel_dataframe
    return dict_of_datasets


def plot_metric(dict_of_datasets, metric, type, acc_list, inference_bool, big=False):
    fig, ax = setup_metric_figure(metric, inference_bool=inference_bool, big=big)
    for ii, i_model in enumerate(MODEL_NAMES):
        model_type = f'{i_model}{type}'
        sel_dataframe = dict_of_datasets[model_type]
        if sel_dataframe is not None:
            for jj, i_acc in enumerate(acc_list):
                sel_acc = sel_dataframe[sel_dataframe['acceleration'] == i_acc]
                mean_key = metric + '_mean'
                if mean_key in sel_acc.keys():
                    mean_values = sel_acc[metric + '_mean']
                    std_values = sel_acc[metric + '_std']
                    plot_bar(ax[jj], x_pos=ii, y_mean=mean_values, y_std=std_values, model_name=i_model)
    #
    for ii, i_ax in enumerate(ax):
        i_ax.set_ylabel(METRIC_NAME_DICT[metric], fontsize=FONTSIZE_YLABEL)
        if legend_bool:
            box = i_ax.get_position()
            i_ax.set_position([box.x0, box.y0 + box.height * 0.1,
                               box.width, box.height * 0.9])
            if ii == 0:
                # Put a legend below current axis
                i_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                            fancybox=True, shadow=True, ncol=len(MODEL_NAMES), fontsize=FONTSIZE_LEGEND)
                # plt.show()
            # _ = i_ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
        # #
        # if ii == 0 and legend_bool:
        #     _ = i_ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)

    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', default=PRETR_SYNTH_ACQ_APPENDIX)
    parser.add_argument('--inference', default=False, action='store_true')
    parser.add_argument('--paper', default=False, action='store_true')
    parser.add_argument('--legend', default=False, action='store_true')

    p_args = parser.parse_args()
    inference_bool = p_args.inference
    sel_type = p_args.type
    paper_bool = p_args.paper
    legend_bool = p_args.legend

    if inference_bool:
        selected_path = DRESULT_INFERENCE
        anatomy_str = 'undersampled'
        acc_list = [0]
        NCOL = 1
        temp_str = 'inference'
    else:
        selected_path = DRESULT
        anatomy_str = 'mixed'
        acc_list = ACCELERATION_LIST
        NCOL = 2
        temp_str = 'test'

    # Only get metrics for models trained with 100p
    dict_of_datasets = get_dict_of_datasets(selected_path, sel_type, anatomy_str, percentage_int=100)
    for i_metric in METRIC_NAMES:
        fig = plot_metric(dict_of_datasets, i_metric, sel_type, acc_list, inference_bool=inference_bool, big=legend_bool)
        if paper_bool:
            if legend_bool:
                dest_path = os.path.join(DRECON, 'Figure 2 - compare models', f'{temp_str}_compare_model_{i_metric}_legend.png')
            else:
                dest_path = os.path.join(DRECON, 'Figure 2 - compare models', f'{temp_str}_compare_model_{i_metric}.png')

        else:
            dest_path = os.path.join(selected_path, f'metric_per_model_100p_{i_metric}_{sel_type}.png')

        if not os.path.isdir(os.path.dirname(dest_path)):
            os.makedirs(os.path.dirname(dest_path))

        if legend_bool:
            fig.savefig(dest_path)
        else:
            fig.savefig(dest_path,  pad_inches=0.0)

        print('Stored to ', dest_path)
        plt.close('all')
