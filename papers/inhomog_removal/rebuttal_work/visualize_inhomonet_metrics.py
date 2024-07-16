import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

"""
here we visualize the InhomoNet metrics and compare it to one of ours. Lets do the single channel t Biasf

Graph is something like this


Bar plot
Synthetic:

    Inhomonet vs model 
    WD/SSIM/RMSE
    
7T


"""

FONTSIZE_XLABEL = 20
FONTSIZE_YLABEL = 20
FONTSIZE_XTICKS = 20
FONTSIZE_YTICKS = 20
FONTSIZE_TITLE = 20
FONTSIZE_LEGEND = 13
FIG_SIZE = (4, 8)
YLIM = 1


def plot_bar_on_ax(ax, x_value, y_value, y_error, color, label):
    ax.bar([x_value], [y_value], yerr=[y_error], width=bar_width, align='center',
           alpha=0.5, color=color,
           ecolor='black',
           label=label,
           capsize=FONTSIZE_TITLE)
    return ax


def get_values(df, metric_list, model_name, data_type):
    return [df.loc[model_name][f'{x}_{data_type}'] for x in metric_list]


def plot_bar(inhomonet_tuple, model_tuple):
    inhom_value, inhom_std = inhomonet_tuple
    model_value, model_std = model_tuple
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax = plot_bar_on_ax(ax, x_value=0, y_value=inhom_value, y_error=inhom_std, color='r', label='Inhomonet')
    ax = plot_bar_on_ax(ax, x_value=bar_width, y_value=model_value, y_error=model_std, color='b', label='Single-channel t-Biasfield')
    #plt.legend(loc='lower right', prop={'size': FONTSIZE_LEGEND})
    ax.set_ylabel(i_metric, fontsize=FONTSIZE_YLABEL)
    # ax.tick_params(axis='x', labelrotation=0, labelsize=FONTSIZE_XTICKS)
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    ax.tick_params(axis='y', labelrotation=0, labelsize=FONTSIZE_YTICKS)
    # ax.set_ylim(0, YLIM)
    ax.set_xticks([bar_width/2])
    # ax.set_xticklabels([i_metric])
    # ax.yaxis.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig


data = {
    'synthetic': ['Uncorrected', 'N4', 'Single biasfield', 'Singe homogeneous', 'Multi biasfield', 'Multi homogeneous', 'Inhomonet'],
    'WD_mean': [0.45, 0.44, 0.08, 0.06, 0.08, 0.06, 0.06],
    'WD_std': [0.03, 0.04, 0.02, 0.01, 0.02, 0.02, 0.02],
    'SSIM_mean': [0.23, 0.24, 0.76, 0.73, 0.77, 0.75, 0.7],
    'SSIM_std': [0.04, 0.05, 0.05, 0.04, 0.05, 0.04, 0.05],
    'RMSE_mean': [9.83, 9.91, 6.33, 6.38, 6.32, 6.28, 6.61],
    'RMSE_std': [0.65, 0.65, 0.54, 0.54, 0.56, 0.51, 0.5]
}

data_singlebiasf = {
    'Single biasfield': ['synthetic', 'volunteer', 'patient', 'patient_3T'],
    'homogeneity_mean': [0.56, 0.39, 0.61, 0.62],
    'homogeneity_std': [0.06, 0.06, 0.05, 0.06],
    'energy_mean': [0.54, 0.36, 0.57, 0.59],
    'energy_std': [0.06, 0.07, 0.06, 0.06],
    'rel_homogeneity_mean': [5.32, 1.51, 0.17, 0.12],
    'rel_homogeneity_std': [1.21, 0.56, 0.05, 0.03],
    'rel_energy_mean': [8.60, 5.71, 0.26, 0.23],
    'rel_energy_std': [2.28, 1.72, 0.06, 0.06]
}


data_inhomonet = {
    'Inhomonet': ['synthetic', 'volunteer', 'patient', 'patient_3T'],
    'homogeneity_mean': [0.57, 0.4, 0.58, 0.62],
    'homogeneity_std': [0.06, 0.06, 0.06, 0.06],
    'energy_mean': [0.54, 0.35, 0.55, 0.59],
    'energy_std': [0.06, 0.07, 0.06, 0.06],
    'rel_homogeneity_mean': [5.42, 1.53, 0.12, 0.13],
    'rel_homogeneity_std': [1.21, 0.56, 0.06, 0.03],
    'rel_energy_mean': [8.58, 5.63, 0.22, 0.22],
    'rel_energy_std': [2.29, 1.73, 0.07, 0.06]
}

DFINAL = '/home/bugger/Documents/paper/inhomogeneity removal/rebuttal'

df = pd.DataFrame(data)
df = df.set_index('synthetic')
metric_list = ['WD', 'SSIM', 'RMSE']

inhomonet_metric_mean = get_values(df, metric_list, 'Inhomonet', 'mean')
inhomonet_metric_std = get_values(df, metric_list, 'Inhomonet', 'std')


model_metric_mean = get_values(df, metric_list, 'Single biasfield', 'mean')
model_metric_std = get_values(df, metric_list, 'Single biasfield', 'std')

bar_width = 0.2
for ii, i_metric in enumerate(metric_list):
    ddest = os.path.join(DFINAL, f'{i_metric}.png')
    inhom_value = inhomonet_metric_mean[ii]
    model_value = model_metric_mean[ii]
    inhom_std = inhomonet_metric_std[ii]
    model_std = model_metric_std[ii]
    fig = plot_bar((inhom_value, inhom_std), (model_value, model_std))
    fig.savefig(ddest, bbox_inches='tight', pad_inches=0.0)


"""
Now create the other one 
"""

df_inhomonet = pd.DataFrame(data_inhomonet)
df_inhomonet = df_inhomonet.set_index('Inhomonet')

df_model = pd.DataFrame(data_singlebiasf)
df_model = df_model.set_index('Single biasfield')

metric_list = ['homogeneity', 'energy']

for selected_dataset in ['synthetic', 'patient']:
    inhomonet_metric_mean = get_values(df_inhomonet, metric_list, selected_dataset, 'mean')
    inhomonet_metric_std = get_values(df_inhomonet, metric_list, selected_dataset, 'std')
    model_metric_mean = get_values(df_model, metric_list, selected_dataset, 'mean')
    model_metric_std = get_values(df_model, metric_list, selected_dataset, 'std')

    bar_width = 0.2
    for ii, i_metric in enumerate(metric_list):
        ddest = os.path.join(DFINAL, f'{i_metric}_{selected_dataset}.png')
        print(ddest)
        inhom_value = inhomonet_metric_mean[ii]
        model_value = model_metric_mean[ii]
        inhom_std = inhomonet_metric_std[ii]
        model_std = model_metric_std[ii]
        fig = plot_bar((inhom_value, inhom_std), (model_value, model_std))
        fig.savefig(ddest, bbox_inches='tight', pad_inches=0.0)

plt.close('all')