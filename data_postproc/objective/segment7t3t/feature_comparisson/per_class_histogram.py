import helper.plot_class as hplotc
import matplotlib.pyplot as plt
import re
import helper.misc as hmisc
import numpy as np
from skimage.util import img_as_ubyte
import small_project.homogeneity_measure.metric_implementations as homog_measure
import helper.array_transf as harray
import scipy.stats
import os
from objective_configuration.segment7T3T import DATASET_LIST, DATASET_SYNTH_LIST
from objective_configuration.segment7T3T import DICT_DATASET_DLABEL, DICT_DATASET_DIMG, DLOG
from data_postproc.objective.segment7t3t.clustering.FileGatherer import FileGatherer


def _update_min_max(min_max_dict, feature_list, feature):
    # Small helper to update min max values
    min_feature = np.min(feature_list)
    max_feature = np.max(feature_list)
    if min_feature < min_max_dict[feature][0]:
        min_max_dict[feature][0] = min_feature
    if max_feature > min_max_dict[feature][1]:
        min_max_dict[feature][1] = max_feature
    return min_max_dict

"""
For all the data that we have.. simply demonstrate the pixel intensities of each class
"""

dpng = '/home/bme001/20184098/visualization'

filegather_object_img = FileGatherer(debug=True, dict_dataset=DICT_DATASET_DIMG)
filegather_object_label = FileGatherer(debug=True, dict_dataset=DICT_DATASET_DLABEL)


# Update on the image files for the 7T dataset so that the label and image files are aligned
base_name_label = [hmisc.get_base_name(x) for x in filegather_object_label.file_dict_list['7t']]

temp_file_list = []
for ifile in filegather_object_img.file_dict_list['7t']:
    stripped_file = re.sub('_0000$', '', hmisc.get_base_name(ifile))
    if stripped_file in base_name_label:
        temp_file_list.append(ifile)

filegather_object_img.file_dict_list['7t'] = temp_file_list

n_img = 100
class_intensities_dataset = {}
list_dataset = DATASET_LIST + DATASET_SYNTH_LIST
n_datasets = len(list_dataset)

for i_dataset in list_dataset:
    _ = class_intensities_dataset.setdefault(i_dataset, {})
    _ = class_intensities_dataset[i_dataset].setdefault('class_1', [])
    _ = class_intensities_dataset[i_dataset].setdefault('class_2', [])
    _ = class_intensities_dataset[i_dataset].setdefault('class_3', [])
    #
    dpng_dataset = os.path.join(dpng, i_dataset)
    if not os.path.isdir(dpng_dataset):
        os.makedirs(dpng_dataset)
    #
    img_array = filegather_object_img.load_n_array(i_dataset)
    label_array = filegather_object_label.load_n_array(i_dataset)
    label_ind = label_array.sum(axis=(-2, -1)) > 0
    img_array = img_array[label_ind]
    label_array = label_array[label_ind]
    collage_obj = hplotc.PlotCollage(content_list=img_array[:3*6], ddest=dpng_dataset, n_display=6, plot_type='img')
    collage_obj.plot_collage()
    collage_masked_obj =hplotc.PlotCollage(content_list=(img_array * (label_array > 0))[:3*6], ddest=dpng_dataset,
                       n_display=6, plot_type='img')
    collage_masked_obj.plot_collage(str_appendix='masked')
    class_1_ind = label_array == (255 // 3)
    class_2_ind = label_array == (255 // 3 * 2)
    class_3_ind = label_array == (255 // 3 * 3)
    pixel_intensities_1 = harray.scale_minmax(img_array[class_1_ind])
    pixel_intensities_2 = harray.scale_minmax(img_array[class_2_ind])
    pixel_intensities_3 = harray.scale_minmax(img_array[class_3_ind])
    class_intensities_dataset[i_dataset]['class_1'].extend(pixel_intensities_1)
    class_intensities_dataset[i_dataset]['class_2'].extend(pixel_intensities_2)
    class_intensities_dataset[i_dataset]['class_3'].extend(pixel_intensities_3)

# Now we have stored all that.. lets visualize it..

ax_shape = hmisc.get_square(n_datasets)
hplotc.close_all()
fig, ax = plt.subplots(*ax_shape, figsize=(20, 10))
ax = ax.ravel()
for ii, i_dataset in enumerate(list_dataset):
    _ = ax[ii].hist(class_intensities_dataset[i_dataset]['class_1'], color='red', label='class_1', alpha=0.3)
    _ = ax[ii].hist(class_intensities_dataset[i_dataset]['class_2'], color='blue', label='class_2', alpha=0.3)
    _ = ax[ii].hist(class_intensities_dataset[i_dataset]['class_3'], color='green', label='class_3', alpha=0.3)
    _ = ax[ii].set_title(i_dataset)
    _ = ax[ii].legend()

plt.tight_layout()
fig.savefig(os.path.join(dpng, 'histogram_individual_classes.png'))

# Another way..

hplotc.close_all()
fig, ax = plt.subplots(3, n_datasets, figsize=(40, 10))
for ii, i_dataset in enumerate(list(class_intensities_dataset.keys())):
    _ = ax[0, ii].hist(class_intensities_dataset[i_dataset]['class_1'], color='red', label='class_1', alpha=0.8)
    _ = ax[1, ii].hist(class_intensities_dataset[i_dataset]['class_2'], color='blue', label='class_2', alpha=0.8)
    _ = ax[2, ii].hist(class_intensities_dataset[i_dataset]['class_3'], color='green', label='class_3', alpha=0.8)
    ax[0, ii].set_title(i_dataset)

plt.tight_layout()
fig.savefig(os.path.join(dpng, 'histogram_individual_classes_2.png'))

hplotc.close_all()
"""
Calculate the Wasserstein distance between each dataset + class

"""


def scale_array(x):
    x = harray.scale_minmax(x)
    x = img_as_ubyte(x)
    return x


def calculate_wasserstein(x, y):
    x = scale_array(x)
    y = scale_array(y)
    hist_x, _ = np.histogram(x.ravel(), bins=256, range=(0, 255))
    hist_y, _ = np.histogram(y.ravel(), bins=256, range=(0, 255))
    wss_distance = scipy.stats.wasserstein_distance(hist_x, hist_y)
    return wss_distance


wss_segmentation_classes = np.zeros((3, n_datasets, n_datasets))
for i_dataset in range(n_datasets):
    i_dataset_name = list_dataset[i_dataset]
    for j_dataset in range(i_dataset, n_datasets):
        j_dataset_name = list_dataset[j_dataset]
        for i_class in [1, 2, 3]:
            x = class_intensities_dataset[i_dataset_name][f'class_{i_class}']
            y = class_intensities_dataset[j_dataset_name][f'class_{i_class}']
            wss_xy = calculate_wasserstein(x, y)
            wss_segmentation_classes[i_class-1, i_dataset, j_dataset] = wss_xy


for i_class in range(3):
    wss_matrix = wss_segmentation_classes[i_class] + wss_segmentation_classes[i_class].T
    wss_matrix[np.diag_indices(wss_matrix.shape[0])] /= 2
    _ = fig_obj = hplotc.ListPlot(np.log(wss_matrix + 1), title=f'class {i_class}', cbar=True)
    _ = fig_obj.ax_list[0].set_yticks(range(len(list_dataset)))
    _ = fig_obj.ax_list[0].set_yticklabels(list_dataset)
    _ = fig_obj.ax_list[0].set_xticklabels(list_dataset, rotation=90)
    _ = fig_obj.ax_list[0].set_xticks(range(len(list_dataset)))
    fig_obj.figure.savefig(os.path.join(dpng, f'log_wasserstein_matrix_class_{i_class}.png'))