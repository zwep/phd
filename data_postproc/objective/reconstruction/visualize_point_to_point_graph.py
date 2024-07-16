import numpy as np
from objective_helper.reconstruction import prepare_metric_dataset, setup_metric_figure
import os
import matplotlib.pyplot as plt
import pandas as pd
from objective_configuration.reconstruction import DRESULT, ACCELERATION_LIST, TYPE_NAMES, MODEL_NAMES, \
    MODEL_COLOR_DICT, METRIC_NAMES, PERCENTAGE_LIST, DRESULT_INFERENCE, MODEL_NAME_DICT, FONTSIZE_YLABEL, METRIC_NAME_DICT

"""
Until now we have made plots of the mean..

What if we draw dots...

Okay we have made it. What do we think of it..? Well, it is not great I guess?


"""


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


selected_path = DRESULT_INFERENCE
anatomy_str = 'undersampled'
acc_list = [0]

dict_of_datasets = get_dict_of_datasets(selected_path, anatomy_str)


sel_model = 'unet_PRETR_SYNTH_ACQ'
sel_metric_str = 'psnr'
sel_metrics = dict_of_datasets[sel_model]
fig, ax = plt.subplots()
remaining_points = True
i_index = 0
while remaining_points:
    perc_metric_list = []
    for i_perc in [0] + PERCENTAGE_LIST:
        temp_value = sel_metrics.loc[sel_metrics['percentage'] == i_perc][sel_metric_str].values[0][i_index]
        perc_metric_list.append(temp_value)
    sel_color = 'r' if i_index % 2 else 'b'
    _ = ax.plot([0] + PERCENTAGE_LIST, perc_metric_list, color = sel_color)
    i_index += 1
    n_total = len(sel_metrics.loc[sel_metrics['percentage'] == i_perc][sel_metric_str].values[0])
    if i_index >= n_total:
        remaining_points = False

fig.savefig(os.path.expanduser('~/test.png'))