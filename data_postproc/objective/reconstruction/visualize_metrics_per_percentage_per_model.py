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
    MODEL_COLOR_DICT, METRIC_NAMES, PERCENTAGE_LIST, DRESULT_INFERENCE, MODEL_NAME_DICT, FONTSIZE_YLABEL, METRIC_NAME_DICT

"""

Compare between percentages and per model

"""


def add_plot_line(ax, x_ticks, y_ticks, y_std, color, label, linewidth=2):
    ax.plot(x_ticks, y_ticks, '-o', color=color, linewidth=linewidth, label=label)
    ax.fill_between(x_ticks, y_ticks - y_std, y_ticks + y_std, color=color, alpha=0.2, edgecolor=color, linewidth=2*linewidth)


def get_dict_of_datasets(selected_path, anatomy_str):
    dict_of_datasets = {}
    for i_type in TYPE_NAMES:
        for i_model in MODEL_NAMES:
            model_type = f'{i_model}{i_type}'
            pred_folder = os.path.join(selected_path, model_type)
            djson = os.path.join(pred_folder, 'metric.json')
            prepped_dataset = prepare_metric_dataset(djson)
            # Make sure we only take the 'mixed' dataset
            if prepped_dataset is not None:
                sel_dataframe = prepped_dataset[prepped_dataset['anatomy'] == anatomy_str]
            else:
                sel_dataframe = prepped_dataset
            dict_of_datasets[model_type] = sel_dataframe
    return dict_of_datasets


def plot_metric(dict_of_datasets, metric, type, selected_path):
    fig, ax = setup_metric_figure(metric)
    sel_percentage = [0] + PERCENTAGE_LIST
    for i_model in MODEL_NAMES:
        model_type = f'{i_model}{type}'
        model_plot_name = MODEL_NAME_DICT[i_model]
        sel_dataframe = dict_of_datasets[model_type]
        if sel_dataframe is not None:
            for ii, i_acc in enumerate(acc_list):
                sel_acc = sel_dataframe[sel_dataframe['acceleration'] == i_acc]
                sel_acc = sel_acc.sort_values('percentage')
                mean_key = metric + '_mean'
                if mean_key in sel_acc.keys():
                    mean_values = sel_acc[metric + '_mean']
                    std_values = sel_acc[metric + '_std']
                    add_plot_line(ax[ii], x_ticks=sel_percentage[-len(mean_values):], y_ticks=mean_values, y_std=std_values,
                                  color=MODEL_COLOR_DICT[i_model], label=model_plot_name)
    #
    for i_ax in ax:
        i_ax.set_ylabel(METRIC_NAME_DICT[metric], fontsize=FONTSIZE_YLABEL)
        _ = i_ax.legend(loc='lower right')

    dest_path = os.path.join(selected_path, f'metric_per_percentage_{metric}_{type}.png')
    if not os.path.isdir(os.path.dirname(dest_path)):
        os.makedirs(os.path.dirname(dest_path))

    fig.savefig(dest_path, bbox_inches='tight', pad_inches=0.0)
    plt.close('all')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', default=False, action='store_true')
    p_args = parser.parse_args()
    inference_bool = p_args.inference

    if inference_bool:
        selected_path = DRESULT_INFERENCE
        anatomy_str = 'undersampled'
        acc_list = [0]
    else:
        selected_path = DRESULT
        anatomy_str = 'mixed'
        acc_list = ACCELERATION_LIST

    dict_of_datasets = get_dict_of_datasets(selected_path, anatomy_str)
    for i_metric in METRIC_NAMES:
        for i_type in TYPE_NAMES:
            plot_metric(dict_of_datasets, i_metric, i_type, selected_path)
