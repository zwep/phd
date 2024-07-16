import getpass

if getpass.getuser() == 'sharreve':
    import matplotlib
    matplotlib.use('Agg')

import os
import seaborn as sns
import matplotlib.pyplot as plt
import helper.array_transf as harray
import pandas as pd
import helper.misc as hmisc
from pathlib import PurePath
import numpy as np
import argparse
from objective_configuration.reconstruction import DRESULT, DRESULT_INFERENCE, METRIC_NAMES
from objective_helper.reconstruction import prepare_metric_dataset

"""
derpederp
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '-m', type=str, help='Model path name, relative to /local_scratch/sharreve/paper/reconstruction/results')
    parser.add_argument('--inference', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')

    # Parses the input
    p_args = parser.parse_args()
    pred_folder = p_args.model
    debug = p_args.debug
    inference_bool = p_args.inference

    if inference_bool:
        pred_folder = os.path.join(DRESULT_INFERENCE, pred_folder)
        anatomy_str = 'undersampled'
    else:
        pred_folder = os.path.join(DRESULT, pred_folder)
        anatomy_str = 'mixed'  # This used to be something useful

    djson = os.path.join(pred_folder, 'metric.json')
    df_reset = prepare_metric_dataset(djson)

    metrics = [x for x in df_reset.columns if 'mean' in x]
    # Create scatter plots using catplot
    for i_metric in METRIC_NAMES:
        if debug:
            print(i_metric)
            print(df_reset)
        if 'ssim' == i_metric:
            max_ylim = 1
        else:
            # Take a maximum + 20%
            max_ylim = df_reset[i_metric + '_mean'].max() * 1.2
        g = sns.catplot(x='percentage', y=i_metric + '_mean', hue='anatomy', col='acceleration',
                    data=df_reset[df_reset['anatomy'] == anatomy_str], kind='swarm', height=5, aspect=1.5)
        n_acc = len(df_reset['acceleration'].value_counts())
        for ii in range(n_acc):
            g.axes[0][ii].set_ylim(0, max_ylim)
        fig = plt.gcf()
        plt.title(f'Mean {i_metric} over all anatomical regions')
        fig.savefig(os.path.join(pred_folder, f'mixed_{i_metric}.png'), bbox_inches='tight', pad_inches=0.0)
