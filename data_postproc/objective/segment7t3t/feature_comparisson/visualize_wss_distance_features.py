import helper.plot_class as hplotc
import time
import pandas as pd
import matplotlib.pyplot as plt
import re
import helper.misc as hmisc
import numpy as np
from skimage.util import img_as_ubyte
import small_project.homogeneity_measure.metric_implementations as homog_measure
import helper.array_transf as harray
import os
from data_postproc.objective.segment7t3t.clustering.FileGatherer import FileGatherer
from objective_configuration.segment7T3T import DATASET_LIST, DATASET_SYNTH_LIST
from objective_configuration.segment7T3T import DICT_DATASET_DLABEL, DICT_DATASET_DIMG, DLOG
import scipy.stats


def calculate_wasserstein(x, y):
    hist_x, _ = np.histogram(x.ravel())
    hist_y, _ = np.histogram(y.ravel())
    wss_distance = scipy.stats.wasserstein_distance(hist_x, hist_y)
    return wss_distance

"""
Here we visualize each feature... and calculate the wasserstein distance between the feature's distributions
"""

dpng = '/home/bme001/20184098/visualization'

list_dataset = DATASET_LIST + DATASET_SYNTH_LIST
n_datasets = len(list_dataset)

# How to get all the features..?
feature_list = pd.read_csv(os.path.join(dpng, '7t', 'image_feature.csv')).columns[1:]

n_img = 100
for sel_feature in feature_list:
# sel_feature = 'homogeneity'
    wss_feature = np.zeros((n_datasets, n_datasets))
    for i_dataset in range(n_datasets):
        i_dataset_name = list_dataset[i_dataset]
        dpng_csv = os.path.join(dpng, i_dataset_name, 'image_feature.csv')
        feature_df_i = np.array(pd.read_csv(dpng_csv)[sel_feature].to_list())
        for j_dataset in range(i_dataset, n_datasets):
            j_dataset_name = list_dataset[j_dataset]
            dpng_csv_j = os.path.join(dpng, j_dataset_name, 'image_feature.csv')
            feature_df_j = np.array(pd.read_csv(dpng_csv_j)[sel_feature].to_list())
            wss_value = np.array(calculate_wasserstein(feature_df_i, feature_df_j))
            wss_feature[i_dataset, j_dataset] = wss_value

    wss_matrix = wss_feature + wss_feature.T
    wss_matrix[np.diag_indices(wss_matrix.shape[0])] /= 2
    _ = fig_obj = hplotc.ListPlot(wss_matrix, title=f'feature {sel_feature}', cbar=True)
    _ = fig_obj.ax_list[0].set_yticks(range(len(list_dataset)))
    _ = fig_obj.ax_list[0].set_yticklabels(list_dataset)
    _ = fig_obj.ax_list[0].set_xticklabels(list_dataset, rotation=90)
    _ = fig_obj.ax_list[0].set_xticks(range(len(list_dataset)))
    fig_obj.figure.savefig(os.path.join(dpng, f'wasserstein_matrix_feature_{sel_feature}.png'))