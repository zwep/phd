"""
The other script `compare_many_T_bart_data.py` created some .csv files with metric values.


"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

ddata = '/home/bugger/Documents/paper/inhomogeneity removal/compare_many_T'

csv_files = [x for x in os.listdir(ddata) if x.endswith('csv')]

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'  : 16}

matplotlib.rc('font', **font)

# Propertie sof the files
plot_keys = ['fsim', 'hpsi', 'jensen_shannon', 'wasserstein']
plot_titles = ['fsi metric', 'hpsi metric', 'Jensen Shannon', 'Wasserstein distance']

# Properties per figure
x_labels = ['n4itk(3T) to 3T', 'n4itk(7T) to 3T', '7T to 3T', 'direct model to 3T', 'biasfield model to 3T']
plot_colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
n_positions = len(csv_files)
for i in range(len(plot_keys)):
    i_key = plot_keys[i]
    i_title = plot_titles[i]

    fig, ax = plt.subplots(figsize=(12, 9))
    positions = np.linspace(0, 1, n_positions)
    boxplot_ax_list = []
    for j, i_file in enumerate(csv_files):
        csv_path = os.path.join(ddata, i_file)
        tmp_pd = pd.read_csv(csv_path, delimiter=",")
        metric_array = list(tmp_pd[i_key])
        tmp = plt.boxplot(metric_array, positions=[positions[j]], patch_artist=True,
                          boxprops=dict(label=x_labels[j], facecolor=plot_colors[j]))
        boxplot_ax_list.append(tmp)


    boxplot_ax_list = [x["boxes"][0] for x in boxplot_ax_list]
    ax.legend(boxplot_ax_list, x_labels, loc='upper right')
    ax.get_xaxis().set_visible(False)
    plt.title(i_title)
    fig.savefig(os.path.join(ddata, i_title + '.png'), bbox_inches='tight')
    plt.close()