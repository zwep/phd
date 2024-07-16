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
    MODEL_COLOR_DICT, METRIC_NAMES, PERCENTAGE_LIST, DRESULT_INFERENCE, MODEL_NAME_DICT, \
    PRETR_SYNTH_APPENDIX, PRETR_SYNTH_ACQ_APPENDIX, SCRATCH_ACQ_APPENDIX, MODEL_COLOR_DICT_DARK, FONTSIZE_YLABEL, \
    PRETR_ACQ_APPENDIX, DRECON, METRIC_NAME_DICT, TYPE_NAME_DICT, FONTSIZE_LEGEND, FONTSIZE_XTICKS, FONTSIZE_YTICKS

"""

Very specialized plotting

I want to see, for each model, a selected metric on the y-axis. On the x-axis the amount of acq training data
and additionally the effect of fine tuning with percentage 7T (acq) data.

This means that we need metrics from _PRETR_ACQ, and _PRETR_SYNTH_ACQ  
"""


def plot_bar(ax, x_pos, y_mean, y_std, label, color, zorder, darkness_factor=0.5):
    z = ax.bar(x_pos, y_mean, width=10, label=label, zorder=zorder, alpha=1, color=color)
    # For now, forget the error-bar stuff. Leave it here in case Alexander thinks otherwise
    # Get the original color
    # original_color = z[0].get_facecolor()
    # Create a darker color by multiplying the original RGB values by the factor
    # darker_color = [c * darkness_factor for c in original_color[:-1]] + [1]
    _ = ax.errorbar(x_pos, y_mean, y_std, lw=2, capsize=5, capthick=2, color=color, alpha=0.5)
    return ax


def get_dict_of_datasets(selected_path, anatomy_str, type_list):
    dict_of_datasets = {}
    for i_type in type_list:
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


def plot_metric(dict_of_datasets, metric, model_name, inference_bool, type_list, add_legend=True):
    fig, ax = setup_metric_figure(metric, x_axis=True, inference_bool=inference_bool, big=add_legend)
    sel_percentage = [0] + PERCENTAGE_LIST
    # Zoiets..?
    for i_ax in ax:
        i_ax.set_ylabel(METRIC_NAME_DICT[metric], fontsize=FONTSIZE_YLABEL)
        i_ax.set_xlabel("amount of training data (%)", fontsize=FONTSIZE_YLABEL)
        i_ax.set_xticks(sel_percentage)
        i_ax.set_xticklabels(sel_percentage)
        i_ax.tick_params(axis='x', labelrotation=0, labelsize=FONTSIZE_XTICKS)
        i_ax.tick_params(axis='y', labelrotation=0, labelsize=FONTSIZE_YTICKS)

    model_plot_name = MODEL_NAME_DICT[model_name]
    model_color_list = [MODEL_COLOR_DICT[model_name], MODEL_COLOR_DICT_DARK[model_name]]
    # Reversing this, so that the legend color is correct
    for ii, i_percentage in enumerate(sel_percentage[::-1]):
        # Order matters here since it is linked to zorder for plotting
        for jj, i_type in enumerate(type_list):
            pos_modifier = 5 if jj == 1 else -5
            model_type = f'{model_name}{i_type}'
            hex_color = model_color_list[jj]
            sel_dataframe = dict_of_datasets[model_type]
            if sel_dataframe is not None:
                for kk, i_acc in enumerate(acc_list):
                    sel_acc = sel_dataframe[sel_dataframe['acceleration'] == i_acc]
                    # if i_type == PRETR_SYNTH_APPENDIX:
                    #     # We need to choose this...
                    #     # Since we train with 100% synthetic data and we want to visualize what 7T data does
                    #     sel_perc = sel_acc[sel_acc['percentage'] == 100]
                    # # else:
                    # not sure anymore if we need to select that 100 percentage
                    sel_perc = sel_acc[sel_acc['percentage'] == i_percentage]
                   # print('model', model_type, sel_perc)
                    mean_key = metric + '_mean'
                    if mean_key in sel_perc.keys():
                        mean_values = sel_perc[metric + '_mean']
                        std_values = sel_perc[metric + '_std']
                        if len(mean_values):
                            if add_legend:
                                ax[kk] = plot_bar(ax[kk], x_pos=i_percentage + pos_modifier, y_mean=mean_values, y_std=std_values,
                                         label=f'{model_plot_name}: {TYPE_NAME_DICT[i_type]}', zorder=jj+5, color=hex_color)
                                # ax[kk].legend(loc='lower right')
                                # legend = ax[kk].legend(loc="lower right", edgecolor="black", fontsize=FONTSIZE_LEGEND, framealpha=1).set_zorder(99)
                                box = ax[kk].get_position()
                                ax[kk].set_position([box.x0, box.y0 + box.height * 0.1,
                                                   box.width, box.height * 0.9])
                                # Put a legend below current axis
                                ax[kk].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                                            fancybox=True, shadow=True, ncol=len(type_list),
                                            fontsize=FONTSIZE_LEGEND)
                            else:
                                plot_bar(ax[kk], x_pos=i_percentage+ pos_modifier, y_mean=mean_values, y_std=std_values,
                                         label=None, zorder=jj+5, color=hex_color)
        add_legend = False

    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', default=False, action='store_true')
    parser.add_argument('--paper', default=False, action='store_true')
    parser.add_argument('--legend', default=False, action='store_true')


    p_args = parser.parse_args()
    inference_bool = p_args.inference
    paper_bool = p_args.paper
    legend_bool = p_args.legend
    print('Legend bool ', legend_bool)
    if inference_bool:
        selected_path = DRESULT_INFERENCE
        anatomy_str = 'undersampled'
        acc_list = [0]
    else:
        selected_path = DRESULT
        anatomy_str = 'mixed'
        acc_list = ACCELERATION_LIST

    # Used for paper
    type_list = [PRETR_SYNTH_ACQ_APPENDIX, PRETR_ACQ_APPENDIX]
    # Used for Alex
    # type_list = [PRETR_SYNTH_ACQ_APPENDIX, SCRATCH_ACQ_APPENDIX]
    dict_of_datasets = get_dict_of_datasets(selected_path, anatomy_str, type_list=type_list)
    print('Got the dicts of the dataset')
    for i_metric in METRIC_NAMES:
        for i_model in ['unet']:
            fig = plot_metric(dict_of_datasets, i_metric, i_model,
                        type_list=type_list,
                        inference_bool=inference_bool,
                        add_legend=legend_bool)
            print('Plotted ', i_metric, i_model)
            ddest = os.path.join(selected_path, f'metric_cross_percentage_7T_{i_model}_{i_metric}.png')
            ddest_legend = os.path.join(selected_path, f'metric_cross_percentage_7T_{i_model}_{i_metric}_legend.png')
            if paper_bool:
                ddest = os.path.join(DRECON, 'Figure 3 - percentage', f'{i_model}_percentage_{i_metric}.png')
                ddest_legend = os.path.join(DRECON, 'Figure 3 - percentage', f'{i_model}_percentage_{i_metric}_legend.png')

            if not os.path.isdir(os.path.dirname(ddest)):
                os.makedirs(os.path.dirname(ddest))

            if legend_bool:
                fig.savefig(ddest_legend)
            else:
                fig.savefig(ddest, bbox_inches='tight', pad_inches=0.0)

            print('Stored at ', ddest)
            # fig.savefig(ddest, bbox_inches='tight', pad_inches=0.0)
            plt.close('all')
