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
    MODEL_COLOR_DICT, METRIC_NAMES, FONTSIZE_XTICKS, FONTSIZE_YTICKS, PERCENTAGE_LIST, DRESULT_INFERENCE, \
    MODEL_NAME_DICT, TYPE_NAME_DICT, TYPE_COLOR_DICT, TYPE_COLOR_DICT_DARK, METRIC_NAME_DICT, DRECON, FONTSIZE_YLABEL, \
    FONTSIZE_LEGEND


"""

Compare between types

"""


def plot_bar(ax, x_pos, y_mean, y_std, model_name, type_name):
    plot_name_type = f'{MODEL_NAME_DICT[model_name]}: {TYPE_NAME_DICT[type_name]}'
    color = TYPE_COLOR_DICT[type_name]
    dark_color = TYPE_COLOR_DICT_DARK[type_name]
    # TYPE_COLOR_DICT
    z = ax.bar(x_pos, y_mean, label=plot_name_type, zorder=5, color=color)
    # # Get the original color
    # original_color = z[0].get_facecolor()
    # # Create a darker color by multiplying the original RGB values by the factor
    # darker_color = [c * darkness_factor for c in original_color[:-1]] + [1]
    _ = ax.errorbar(x_pos, y_mean, y_std, lw=2, capsize=5, capthick=2, color=dark_color)
    return ax


def get_dict_of_datasets(selected_path, anatomy_str, type_list, model_name):
    # Get the metrics dicts of all types belonging to a selected model (model_name)
    dict_of_datasets = {}
    for i_type in type_list:
        model_type = f'{model_name}{i_type}'
        pred_folder = os.path.join(selected_path, model_type)
        djson = os.path.join(pred_folder, 'metric.json')
        prepped_dataset = prepare_metric_dataset(djson)
        # Make sure we only take the 'mixed' dataset
        sel_dataframe = prepped_dataset[prepped_dataset['anatomy'] == anatomy_str]
        dict_of_datasets[model_type] = sel_dataframe
    return dict_of_datasets


# Okay here we go....
# model_name = 'unet'
#model_name = 'xpdnet'
#  = ['SCRATCH_SYNTH', 'SCRATCH_ACQ', 'SCRATCH_SYNTH_ACQ', 'PRETR_SYNTH', 'PRETR_ACQ',  'PRETR_SYNTH_ACQ']

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default=None,
                        type=str,
                        nargs='?',  # This makes it optional
                        help='Provide the name of the model that we want to 3 process')
    parser.add_argument('--inference', default=False, action='store_true')
    parser.add_argument('--paper', default=False, action='store_true')
    parser.add_argument('--legend', default=False, action='store_true')

    p_args = parser.parse_args()
    model_name = p_args.model
    inference_bool = p_args.inference
    paper_bool = p_args.paper
    legend_bool = p_args.legend

    if inference_bool:
        selected_path = DRESULT_INFERENCE
        anatomy_str = 'undersampled'
        acc_list = [0]
        NCOL = 1
    else:
        selected_path = DRESULT
        anatomy_str = 'mixed'
        acc_list = ACCELERATION_LIST
        NCOL = 2
        #

    dict_of_datasets = get_dict_of_datasets(selected_path=selected_path, anatomy_str=anatomy_str,
                                            type_list=TYPE_NAMES, model_name=model_name)

    # Add type colors..
    for sel_metric in METRIC_NAMES:
        fig, ax = setup_metric_figure(metric=sel_metric, inference_bool=inference_bool, big=legend_bool)
        for ii, i_acc in enumerate(acc_list):
            print('Acceleration ', i_acc)
            for jj, i_type in enumerate(TYPE_NAMES):
                print('Model type ', i_type)
                model_type = f'{model_name}{i_type}'
                sel_dataframe = dict_of_datasets[model_type]
                sel_acc = sel_dataframe[sel_dataframe['acceleration'] == i_acc]
                sel_perc = sel_acc[sel_acc['percentage'] == 100]
                sel_perc = sel_perc.sort_values('percentage')
                if sel_metric + '_mean' in sel_perc.keys():
                    mean_values = sel_perc[sel_metric + '_mean']
                    std_values = sel_perc[sel_metric + '_std']
                    plot_bar(ax[ii], jj, mean_values, std_values, model_name=model_name, type_name=i_type)

        for ii, i_ax in enumerate(ax):
            i_ax.set_ylabel(METRIC_NAME_DICT[sel_metric], fontsize=FONTSIZE_YLABEL)
            # Only set legend for fist axis
            if legend_bool:
                box = i_ax.get_position()
                i_ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                   box.width, box.height * 0.9])
                if ii == 0:
                    # Put a legend below current axis
                    i_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                              fancybox=True, shadow=True, ncol=len(TYPE_NAMES), fontsize=FONTSIZE_LEGEND)
                    #plt.show()
                # _ = i_ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
        #
        dest_path = os.path.join(selected_path, f'metric_per_type_{model_name}_' + f'{sel_metric}_' + '_'.join(TYPE_NAMES) + '.png')
        if paper_bool:
            if inference_bool:
                temp_str = 'inference'
            else:
                temp_str = 'test'
            if legend_bool:
                dest_path = os.path.join(DRECON, 'Figure 1 - compare types', f'{temp_str}_{model_name}_{sel_metric}_legend.png')
            else:
                dest_path = os.path.join(DRECON, 'Figure 1 - compare types', f'{temp_str}_{model_name}_{sel_metric}.png')

        if not os.path.isdir(os.path.dirname(dest_path)):
            os.makedirs(os.path.dirname(dest_path))

        print('Storing ', dest_path)
        if legend_bool:
            fig.savefig(dest_path)
        else:
            # fig.savefig(dest_path, bbox_inches='tight', pad_inches=0.0)
            fig.savefig(dest_path, pad_inches=0.0)


