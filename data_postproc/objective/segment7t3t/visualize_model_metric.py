import sys

import helper.array_transf

sys.path.append('/home/bugger/PycharmProjects/pytorch_in_mri')
import argparse
import json
import helper.array_transf as harray
import helper.misc as hmisc
import objective_helper.segment7T3T as hsegm7t
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageColor
from objective_configuration.segment7T3T import CLASS_INTERPRETATION, COLOR_DICT, \
    COLOR_DICT_RGB, CMAP_ALL, MY_CMAP, get_path_dict, TRANSFORM_MODEL_NAMES, DLOG
from loguru import logger

logger.add(os.path.join(DLOG, 'visualize_model_metric.log'))
logger.debug("\t\t ==== Starting visualize model modetric === ")

updated_names = {'501': 'No augmentation',  # M&Ms 1 A
                 '502': 'No augmentation',  # M&Ms 1 B
                 '503': 'No augmentation',  # M&Ms 1 B
                 '511': 'No augmentation',  # ACDC
                 '513': 'No augmentation',  # Kaggle
                 '514': 'No augmentation'}  # M&Ms 2}


def set_options_axis(plot_axis):
    plot_axis.legend(loc='lower right', prop={'size': LEGEND_SIZE})
    plot_axis.set_ylim(YLIM_RANGE)
    plot_axis.tick_params(axis='y', which='major', labelsize=FONTSIZE_YTICKS)
    plot_axis.set_xlabel('')
    plot_axis.set_ylabel(YLABEL, fontsize=FONTSIZE_YLABEL)
    plot_axis.grid(zorder=0, color='gray')
    return plot_axis


def translate_xticks_label(old_label):
    new_label = []
    for i_label in old_label:
        found = False
        for i_key, i_value in TRANSFORM_MODEL_NAMES.items():
            if i_key in i_label.lower():
                new_label.append(i_value)
                found = True
        if found is False:
            new_label.append(i_label)
    return new_label


def get_fig_boxplot():
    fig, ax = plt.subplots(figsize=FIGSIZE_BOXPLOT)
    ax = set_axis_width(ax)
    ax.set_xlabel('')
    ax.tick_params(axis='y', which='major', labelsize=FONTSIZE_YTICKS)
    ax.set_ylabel(column_name, fontsize=FONTSIZE_YLABEL)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.8)
    ax.set_ylim(YLIM_RANGE)
    return fig, ax


def set_axis_width(ax, width=2):
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(width)
    # increase tick width
    ax.tick_params(width=width)
    return ax


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str)
parser.add_argument('-model', type=str, default=None)
parser.add_argument('-metric', type=str, default=None)
parser.add_argument('-labels', type=bool, default=True)
parser.add_argument('-ylim', type=str, default=None)
parser.add_argument('-figsize_boxplot', type=str, default=None)
parser.add_argument('-update_names', type=str, default=False)


p_args = parser.parse_args()
dataset = p_args.dataset
change_labels = p_args.labels
model_selection = p_args.model
metric = p_args.metric
ylim = p_args.ylim
figsize_boxplot = p_args.figsize_boxplot
update_names = p_args.update_names

if update_names:
    # We dont need to have the name here...
    TRANSFORM_MODEL_NAMES.update(updated_names)


path_dict = get_path_dict(dataset)
ddest = path_dict['dresults']

if metric in ['hausdorf', 'hd', 'dorfee']:
    ddata = path_dict['dhausdorf']
    column_name = 'hausdorf'
    YLABEL = 'Hausdorff Distance'
elif metric in ['dice', 'dicescore', 'dc', 'd']:
    ddata = path_dict['ddice']
    column_name = 'dice score'
    YLABEL = 'Dice Score'
elif metric in ['jaccard', 'jc', 'j']:
    ddata = path_dict['djaccard']
    column_name = 'jaccard'
    YLABEL = 'Jaccard Score'
else:
    logger.debug('Unknown metrics selected: ', metric)
    logger.debug("Please choose: hd (hausdorf distance), dc (dice score), jc (jaccard score")
    sys.exit()


with open(ddata, 'r') as f:
    temp = f.read()

model_output = json.loads(temp)

data_frame_results = harray.nested_dict_to_df(model_output, column_name=column_name)
data_frame_results = data_frame_results.rename_axis(["Model", "Subject", "Class", "Phase"])

## Write a function from this..
model_name_list = np.array(sorted(list(data_frame_results.index.levels[0])))
logger.debug("List of model names:")
for i, imodelname in enumerate(model_name_list):
    logger.debug(str(i), '\t', imodelname)

import objective_helper.segment7T3T as hsegm7t
if model_selection:
    sel_model_name_list = hsegm7t.model_selection_processor(model_selection, model_name_list)
    data_frame_results = data_frame_results.loc[sel_model_name_list]
else:
    logger.debug("Please select a model first..")
    sys.exit()


logger.debug("Selected model name list")
logger.debug(sel_model_name_list)
logger.debug("Example of ordering of data frame ")
logger.debug(data_frame_results)
# Aggregate stuff
aggr_model_subject_class = data_frame_results.groupby(by=["Model", "Subject", "Class"], sort=False).mean()
aggr_model_subject = data_frame_results.groupby(by=["Model", "Subject"], sort=False).mean()
# Group over all model and classes and make sure that the ordering is preserved
aggr_model_class = data_frame_results.groupby(by=["Model", "Class"], sort=False).mean()
aggr_model_class_unstack = aggr_model_class.unstack(level=1).loc[sel_model_name_list]
# Aggregate over all the models. Somehow sort is not respected.
aggr_model = data_frame_results.groupby(by=["Model"], sort=False).mean()
aggr_model = aggr_model.loc[sel_model_name_list]

"""
Setup for the plots...
"""

# Setup for figures..
LEGEND_SIZE = 20
FONTSIZE_XTICKS = 30
FONTSIZE_YTICKS = 30
FONTSIZE_YLABEL = 30
ROTATION_X_LABEL = 90

if 'haus' in column_name:
    hd_array = np.array(data_frame_results['hausdorf'])
    hd_array = helper.array_transf.correct_inf_nan(hd_array)
    mean_hd = hd_array.mean()
    std_hd = hd_array.std()
    YLIM_RANGE = (0, mean_hd + 3 * std_hd)
else:
    YLIM_RANGE = (0, 1.05)

if ylim is not None:
    ylim = int(ylim)
    YLIM_RANGE = (0, ylim)


BOXPLOT_WIDTH = 0.2
if figsize_boxplot is None:
    FIGSIZE_BOXPLOT = (10, 15)
else:
    # Should be something like 10.15
    x_size, y_size = map(int, figsize_boxplot.split('.'))
    FIGSIZE_BOXPLOT = (x_size, y_size)

FIGSIZE_BARPLOT = (20, 15)
BOXPROPS = dict(linestyle='-', linewidth=2, color='#1B1A17', facecolor='#E45826')
MEDIANPROPS = dict(linestyle='-', linewidth=2, color='#F0A500')

"""
Create the box plot per model 
"""

fig, ax = get_fig_boxplot()
counter = -BOXPLOT_WIDTH*1.1
position_label_list = []
boxplot_axes = []
groupby_obj = data_frame_results.groupby(by=["Model"], sort=False)
max_whisker = 1
for model_name in sel_model_name_list:
    igroup = groupby_obj.get_group(model_name)
    igroup.columns = [column_name]
    counter += BOXPLOT_WIDTH*1.1
    position_label_list.append((counter, model_name))
    data_to_plot = igroup[column_name]
    data_to_plot.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_to_plot.dropna(how="all", inplace=True)
    ax_obj = ax.boxplot(data_to_plot, positions=[counter], showfliers=False, widths=BOXPLOT_WIDTH,
                        patch_artist=True, boxprops=BOXPROPS, medianprops=MEDIANPROPS)
    temp_max_whisker = np.array([x.get_xydata() for x in ax_obj['whiskers']]).max()
    if temp_max_whisker > max_whisker:
        max_whisker = temp_max_whisker
    boxplot_axes.append(ax_obj)

# Convert any labels to a new form
xticks, x_labels = zip(*position_label_list)
x_labels_new = translate_xticks_label(x_labels)

ax.set_xlim((xticks[0]-BOXPLOT_WIDTH/2-0.05, xticks[-1]+BOXPLOT_WIDTH/2+0.05))
if change_labels:
    ax.set_xticks(ticks=xticks, labels=x_labels_new, rotation=ROTATION_X_LABEL, fontsize=FONTSIZE_XTICKS)
else:
    ax.set_xticks(ticks=xticks, labels=x_labels, rotation=ROTATION_X_LABEL, fontsize=FONTSIZE_XTICKS)

ax = set_options_axis(ax)

if 'haus' in column_name:
    ax.legend([boxplot_axes[0]["boxes"][0]], ['column_name'], loc='upper right', prop={'size': LEGEND_SIZE})
    if ylim is None:
        ax.set_ylim((0, max_whisker * 1.05))
else:
    ax.legend([boxplot_axes[0]["boxes"][0]], ['column_name'], loc='lower right', prop={'size': LEGEND_SIZE})

fig.savefig(os.path.join(ddest, f"{column_name}_per_model_boxplot.png"), bbox_inches='tight')

"""
Create boxplot per model per class
"""

fig, ax = get_fig_boxplot()

counter = -BOXPLOT_WIDTH * 1.3
new_xticks_label = []
boxplot_axes = []
median_value_dict = {}
mean_value_dict = {}
groupby_model_class_obj = data_frame_results.groupby(by=["Model", "Class"], sort=False)
max_whisker = 1
for model_name in sel_model_name_list:
    IQR_list = []
    for class_id in range(1, 4):
        class_id = str(class_id)
        igroup = groupby_model_class_obj.get_group((model_name, class_id))
        igroup.columns = [column_name]
        print(model_name, " class ", class_id)
        if class_id != '1':
            counter += BOXPLOT_WIDTH
        else:
            counter += BOXPLOT_WIDTH*1.3
            new_xticks_label.append((counter+BOXPLOT_WIDTH/2, model_name))
        data_to_plot = igroup[column_name]
        data_to_plot.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_to_plot.dropna(how="all", inplace=True)
        ax_obj = ax.boxplot(data_to_plot, positions=[counter], showfliers=False, widths=BOXPLOT_WIDTH,
                            patch_artist=True, boxprops=BOXPROPS, medianprops=MEDIANPROPS)
        IQR = abs(np.diff([item.get_ydata()[0] for item in ax_obj['whiskers']]))
        IQR_list.append(IQR[0])
        logger.debug(f'{model_name} - {class_id} - IQR {IQR}')
        temp_max_whisker = np.array([x.get_xydata() for x in ax_obj['whiskers']]).max()
        if temp_max_whisker > max_whisker:
            max_whisker = temp_max_whisker
        ax_obj['boxes'][0].set_facecolor(COLOR_DICT[class_id])
        median_value = data_to_plot.median()
        mean_value = np.round(data_to_plot.mean(), 2)
        median_value_dict.setdefault(model_name, [])
        median_value_dict[model_name].append(median_value)
        mean_value_dict.setdefault(model_name, [])
        mean_value_dict[model_name].append(mean_value)
        boxplot_axes.append(ax_obj)
    mean_IQR = np.mean(IQR_list)
    logger.debug(f'Dataset: {dataset} Model name: {model_name} mean IQR {mean_IQR}')

xticks, x_labels = zip(*new_xticks_label)
x_labels_new = translate_xticks_label(x_labels)

ax.set_xlim((xticks[0]-BOXPLOT_WIDTH-0.05, xticks[-1]+2*BOXPLOT_WIDTH+0.05))
if change_labels:
    ax.set_xticks(ticks=xticks, labels=x_labels_new, rotation=ROTATION_X_LABEL, fontsize=FONTSIZE_XTICKS)
else:
    ax.set_xticks(ticks=xticks, labels=x_labels, rotation=ROTATION_X_LABEL, fontsize=FONTSIZE_XTICKS)

ax = set_options_axis(ax)
if 'haus' in column_name:
    ax.legend([x["boxes"][0] for x in boxplot_axes], ['RV', 'MYO', 'LV'], loc='upper right', prop={'size': LEGEND_SIZE})
    if ylim is None:
        ax.set_ylim((0, max_whisker * 1.05))
else:
    ax.legend([x["boxes"][0] for x in boxplot_axes], ['RV', 'MYO', 'LV'], loc='lower right', prop={'size': LEGEND_SIZE})

fig.savefig(os.path.join(ddest, f"{column_name}_per_model_per_class_boxplot.png"), bbox_inches='tight')


"""
Print numeric values to output
"""

for iname, igroup in data_frame_results.groupby(by=["Model"], sort=False):
    print(iname, np.round(igroup.mean().values[0], 2))

model_names = list(median_value_dict.keys())
for ii in range(len(model_names)):
    for jj in range(ii, len(model_names)):
        x_mean = np.array(mean_value_dict[model_names[ii]])
        y_mean = np.array(mean_value_dict[model_names[jj]])
        print(model_names[ii], "\t", model_names[jj])
        # Average improvement per class
        print("\t", y_mean, x_mean, np.round(np.mean(x_mean), 2), " - ",
              np.round(np.mean(y_mean), 2), "\t",
              np.round(y_mean / x_mean, 2), "\t",
              np.round(np.mean(y_mean / x_mean), 2), "\t",
              np.round(np.mean((y_mean-x_mean) / x_mean), 2), "\t",
              np.round(np.mean(y_mean) / np.mean(x_mean), 2), "\t",
              np.round(np.mean(y_mean[1:]) / np.mean(x_mean[1:]), 2), "\t")

"""
Create the barplots
"""

aggr_model.columns = [column_name]
x_labels = list(aggr_model.index)
x_labels_new = translate_xticks_label(x_labels)

# Create plots for dice score per model
temp_ax = aggr_model.plot.bar(figsize=FIGSIZE_BARPLOT, facecolor='#E45826')
if change_labels:
    temp_ax.set_xticks(ticks=range(len(x_labels_new)), labels=x_labels_new, rotation=ROTATION_X_LABEL, fontsize=FONTSIZE_XTICKS)
else:
    temp_ax.set_xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=ROTATION_X_LABEL, fontsize=FONTSIZE_XTICKS)

temp_ax = set_options_axis(temp_ax)
temp_ax.figure.savefig(os.path.join(ddest, f"{column_name}_per_model.png"), bbox_inches='tight')

# Create plots for dice score per model per class
aggr_model_class_unstack.columns = [CLASS_INTERPRETATION[x] for x in aggr_model_class_unstack.columns.get_level_values(1)]
temp_ax = aggr_model_class_unstack.plot.bar(figsize=FIGSIZE_BARPLOT, color=COLOR_DICT.values())
if change_labels:
    temp_ax.set_xticks(ticks=range(len(x_labels_new)), labels=x_labels_new, rotation=ROTATION_X_LABEL, fontsize=FONTSIZE_XTICKS)
else:
    temp_ax.set_xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=ROTATION_X_LABEL, fontsize=FONTSIZE_XTICKS)

temp_ax = set_options_axis(temp_ax)
temp_ax.figure.savefig(os.path.join(ddest, f"{column_name}_per_model_per_class.png"), bbox_inches='tight')


"""
Visualize single class....
- a bit hacky,, but sure
"""

if False:
    # Only do this when we will be using the standard apporach
    nbins = 32
    num_groups = len(aggr_model.index)
    # cmap = matplotlib.cm.get_cmap('cool', lut=num_groups)
    color_synth = np.array(ImageColor.getcolor("#F4BFBF", "RGB")) / 256
    color_baseline = np.array(ImageColor.getcolor("#FFD9C0", "RGB")) / 256
    color_biasf = np.array(ImageColor.getcolor("#8CC0DE", "RGB")) / 256
    color_array = np.array([color_baseline, color_biasf, color_synth])
    my_cmap = ListedColormap(color_array)
    fig, axes = plt.subplots(num_groups)
    axes_mapping = {'synth-model-31-03': [2, 'Synthetic'],
                    'Task511_ACDC': [0, 'Baseline'],
                    'Task612_ACDC_Biasfield_ACDC': [1, 'Biasfield']}
    for i, (k, group) in enumerate(data_frame_results.groupby(by=["Model", "Class"], sort=False)):
        if group.iloc[0].name[2] != '1':
            continue
        model_name, class_name = k
        print(k)
        useful_index, model_label = axes_mapping[model_name]
        counts, value, _ = axes[useful_index].hist(group.values, histtype='bar', bins=nbins, label=model_label, color=my_cmap(useful_index), linewidth=1, edgecolor='white')
        #
        print("Average", np.sum(counts * value[:-1]) / np.sum(counts))
        axes[useful_index].text(value[0] + 1 / (4 * nbins), counts[0], f'{int(counts[0])}')
        axes[useful_index].text(value[-3] + 1 / (4 * nbins), counts[-2], f'{int(counts[-2])}')
        axes[useful_index].legend(loc='upper right')
        axes[useful_index].set_ylim(0, max(counts)+5)
        axes[useful_index].set_xlim(-0.05, 1.05)
        new_yticks = np.linspace(min(counts), max(counts)+5, 5, dtype=int)
        axes[useful_index].set_yticks(ticks=new_yticks)
        axes[useful_index].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.8)
        #
    fig.savefig(os.path.join(ddest, f"{column_name}_distribution_RV.png"), bbox_inches='tight')

"""
Store all CSVs (containing aggregated results)
"""
temp_name = os.path.join(ddest, f"{column_name}_per_model_subject_class_phase.csv")
data_frame_results.to_csv(temp_name, sep=",", index=True)

temp_name = os.path.join(ddest, f"{column_name}_per_model_subject_class.csv")
aggr_model_subject_class.to_csv(temp_name, sep=",", index=True)

temp_name = os.path.join(ddest, f"{column_name}_per_model_subject.csv")
aggr_model_subject.to_csv(temp_name, sep=",", index=True)

temp_name = os.path.join(ddest, f"{column_name}_per_model_class.csv")
aggr_model_class.to_csv(temp_name, sep=",", index=True)

temp_name = os.path.join(ddest, f"{column_name}_per_model.csv")
aggr_model.to_csv(temp_name, sep=",", index=True)

