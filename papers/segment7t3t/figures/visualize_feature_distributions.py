import matplotlib.pyplot as plt
import os
import pandas as pd
from objective_configuration.segment7T3T import DATASET_LIST, DATASET_SYNTH_LIST
import numpy as np
import helper.misc as hmisc

"""
visualize two features and try to see if we can cluster the datasets based on that?
"""

dpng = '/home/bme001/20184098/visualization'


class PlotFeatures:
    FONT_SIZE = 40
    FONT_SIZE_LEGEND = 30
    FONT_SIZE_XTICKS = 30
    FONT_SIZE_YTICKS = 30
#
    NAME_CHANGER = {'contrast': 'Contrast', 'homogeneity': 'Homogeneity',
                    'energy': 'Energy', 'std': 'Standard deviation',
                    'mean': 'Mean', 'LV': 'Left ventricle', 'RV': 'Right ventricle',
                    'MYO': 'Myocardium',
                    'All': 'Full heart', 'body': 'Full image'}

    def __init__(self, dataset_list, feature_list=None):
        self.dataset_list = dataset_list
        self.n_datasets = len(dataset_list)
        if feature_list is None:
            self.feature_list = ['contrast', 'homogeneity', 'energy', 'std', 'mean']
        else:
            self.feature_list = feature_list
        self.n_feature = len(self.feature_list)
#
    def plot_features(self, classes, nrows, figsize=None):
        if figsize is None:
            figsize = (30, 20)
        fig, ax = plt.subplots(nrows=nrows, ncols=self.n_feature, figsize=figsize)
        if ax.ndim == 1:
            ax = ax[None]
        #
        for i, i_feature in enumerate(self.feature_list):
            for i_dataset in range(self.n_datasets):
                i_dataset_name = list_dataset[i_dataset]
                dpng_csv = os.path.join(dpng, i_dataset_name, 'image_feature.csv')
                temp_df = pd.read_csv(dpng_csv)
                temp_df = temp_df.set_index(['filename', 'class'])
                for jj, i_class in enumerate(classes):
                    feature_values = temp_df.loc[(slice(None), [i_class]), i_feature].tolist()
                    feature_values = np.array(feature_values)[~np.isnan(feature_values)]
                    bins = hmisc.get_freedman_bins(feature_values)
                    _ = ax[jj, i].hist(feature_values, label=i_dataset_name.upper(), alpha=0.5, bins=bins)
                    _ = ax[jj, i].legend(fontsize=self.FONT_SIZE_LEGEND)
                    ax[jj, i].tick_params(axis='x', which='major', labelsize=self.FONT_SIZE_XTICKS)
                    ax[jj, i].tick_params(axis='y', which='major', labelsize=self.FONT_SIZE_YTICKS)
                    if jj == 0:
                        _ = ax[jj, i].set_title(f'{self.NAME_CHANGER[i_feature]}', fontsize=self.FONT_SIZE)
                    if i == 0:
                        _ = ax[jj, i].set_ylabel(f'{self.NAME_CHANGER[i_class]}', fontsize=self.FONT_SIZE)
        return fig, ax


if __name__ == "__main__":
    # This is the full dataset.
    # list_dataset = DATASET_LIST + DATASET_SYNTH_LIST
    # Here I only select 7T, ACDC, MM1A and MM1B
    list_dataset = DATASET_LIST[:4]
    classes_body = ['body']
    classes_all = ['All']
    classes_segm = ['LV', 'MYO', "RV"]
    plot_obj = PlotFeatures(dataset_list=list_dataset)
    fig1, _ = plot_obj.plot_features(nrows=1, classes=classes_all)
    fig1.savefig(os.path.join(dpng, '2d_feature_plot', f'7T_compare_feature_heart.png'))
    fig2, _ = plot_obj.plot_features(nrows=1, classes=classes_body)
    fig2.savefig(os.path.join(dpng, '2d_feature_plot', f'7T_compare_feature_full_image.png'))
    fig3, _ = plot_obj.plot_features(nrows=3, classes=classes_segm)
    fig3.savefig(os.path.join(dpng, '2d_feature_plot', f'7T_compare_feature_LV_MYO_RV.png'))
    fig4, _ = plot_obj.plot_features(nrows=1, classes=classes_segm[0:1])
    fig4.savefig(os.path.join(dpng, '2d_feature_plot', f'7T_compare_feature_LV.png'))
    fig5, _ = plot_obj.plot_features(nrows=1, classes=classes_segm[1:2])
    fig5.savefig(os.path.join(dpng, '2d_feature_plot', f'7T_compare_feature_MYO.png'))
    fig6, _ = plot_obj.plot_features(nrows=1, classes=classes_segm[2:3])
    fig6.savefig(os.path.join(dpng, '2d_feature_plot', f'7T_compare_feature_RV.png'))