import matplotlib.pyplot as plt
import os
import pandas as pd
from objective_configuration.segment7T3T import DATASET_LIST, DATASET_SYNTH_LIST
import helper.plot_class as hplotc
import numpy as np
import helper.misc as hmisc
import helper.array_transf as harray
import itertools

"""
visualize two features and try to see if we can cluster the datasets based on that?
"""

dpng = '/home/bme001/20184098/visualization'

list_dataset = DATASET_LIST + DATASET_SYNTH_LIST
n_datasets = len(list_dataset)

# How to get all the features..?
feature_list = list(pd.read_csv(os.path.join(dpng, '7t', 'image_feature_classes.csv'), skiprows=1).columns[2:])
product_feature_list = []
n_feature = len(feature_list)
for i_feature in range(n_feature):
    for j_feature in range(i_feature + 1, n_feature):
        temp_prod = (feature_list[i_feature], feature_list[j_feature])
        product_feature_list.append(temp_prod)


for feature_x, feature_y in product_feature_list:
    fig, ax = plt.subplots()
    for i_dataset in range(n_datasets):
        i_dataset_name = list_dataset[i_dataset]
        dpng_csv = os.path.join(dpng, i_dataset_name, 'image_feature.csv')
        feature_x_values = np.array(pd.read_csv(dpng_csv)[feature_x].to_list())
        feature_y_values = np.array(pd.read_csv(dpng_csv)[feature_y].to_list())
        ax.scatter(feature_x_values, feature_y_values, label=i_dataset_name, alpha=0.5)
    #
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.legend()
    fig.savefig(os.path.join(dpng, '2d_feature_plot', f'{feature_x}_{feature_y}_2d_feature_plot.png'))
