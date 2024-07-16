import itertools
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import collections
import os
import helper.misc as hmisc
import re
import numpy as np
import skimage.transform as sktransform
import helper.array_transf as harray
from skimage.util import img_as_ubyte
from skimage.metrics import structural_similarity
import scipy.stats
import helper.metric as hmetric
import helper.plot_class as hplotc

# New method..
import umap
from sklearn.preprocessing import StandardScaler


def plot_embedding(embedding, row_label_dict):
    n_clusters = len(row_label_dict)
    cmap = matplotlib.cm.get_cmap('plasma', lut=n_clusters)
    x0 = 0
    x1 = 0
    prev_amount = 0
    fig, ax = plt.subplots()
    for ii, (ikey, iamount) in enumerate(row_label_dict.items()):
        x0 += prev_amount
        x1 += iamount
        ax.scatter(embedding[x0:x1, 0], embedding[x0:x1, 1], color=cmap(ii), label=ikey)
        prev_amount = iamount
    ax.legend()
    return fig


def upper_tri_to_full(x):
    x_full = x + x.T
    x_full[np.diag_indices(x_full.shape[0])] /= 2
    return x_full


def get_eucledian_distance(X):
    # Needed this, because the built-in Eucledian distance matrix of MDS was not adequate
    # For specific VGG maps. It is faster though
    n_points, n_dim = X.shape
    dist_matrix = np.zeros((n_points, n_points))
    for ii in range(n_dim):
        dist_matrix += ((X[:, ii:ii + 1] - X[:, ii:ii + 1].T)) ** 2
    dist_matrix = np.sqrt(dist_matrix)
    return dist_matrix


ddata = '/data/seb/data/similarity_matrix'
ddata_plots = '/data/seb/data/similarity_matrix/cluster_img'
# Load the VGG-feature maps
dict_vgg = {}
for i_layer in range(4):
    temp_vgg = np.load(os.path.join(ddata, f'pca_feature_array_vgg_{i_layer}.npy'))
    dict_vgg[f'vgg_{i_layer}'] = temp_vgg

# Load the VGG-metric data
dict_vgg_metric = {}
for i_layer in range(4):
    contrast_name = f'contrast_matrix_vgg_{i_layer}'
    ssim_name = f'ssim_matrix_vgg_{i_layer}'
    wss_name = f'wss_matrix_vgg_{i_layer}'
    contrast_vgg = np.load(os.path.join(ddata, contrast_name + '.npy'))
    ssim_vgg = np.load(os.path.join(ddata, ssim_name + '.npy'))
    wss_vgg = np.load(os.path.join(ddata, wss_name + '.npy'))
    # DOnt I need to change the contrast too?
    dict_vgg_metric[contrast_name] = upper_tri_to_full(contrast_vgg)
    # Change SSIM to dissimilarity..
    dict_vgg_metric[ssim_name] = (1-upper_tri_to_full(ssim_vgg)) / 2
    dict_vgg_metric[wss_name] = upper_tri_to_full(wss_vgg)


# Load the id behind the data points...
with open(os.path.join(ddata, 'key_length_string.txt'), 'r') as f:
    temp_read = f.read()

temp_parsed = [x.split(":") for x in temp_read.split("\n")]

ssim_row_id_dict = OrderedDict()
for i_item in temp_parsed:
    if len(i_item) > 1:
        x_key, x_value = i_item
        ssim_row_id_dict.update({x_key: int(x_value)})

label_array = []
for k, v in ssim_row_id_dict.items():
    label_array.extend([k] * v)

"""
Clustering using the featuremaps themselves..
"""

# Using MDS on Eucledian distance VGG maps..
for x_key, x_vgg in dict_vgg.items():
    mds = MDS(dissimilarity='precomputed', n_init=10)
    n_points = x_vgg.shape[0]
    x_vgg = x_vgg.reshape((n_points, -1))
    x_eucledian_dist = get_eucledian_distance(x_vgg)
    # Somehow the result of Eucledian Distance might not be symmetric..
    from sklearn.metrics.pairwise import euclidean_distances
    X_transform = mds.fit_transform(x_eucledian_dist)
    fig_obj = plot_embedding(X_transform, ssim_row_id_dict)
    fig_obj.savefig(os.path.join(ddata_plots, f'MDS_VGG_{x_key}.png'))


# Using Isomap
n_neigh = 5
for x_key, x_vgg in dict_vgg.items():
    # Try clustering based on Multidimensional Scaling...
    n_points = x_vgg.shape[0]
    x_vgg = x_vgg.reshape((n_points, -1))
    ismap_obj = Isomap(n_neighbors=n_neigh)
    X_transform = ismap_obj.fit_transform(x_vgg)
    fig_obj = plot_embedding(X_transform, ssim_row_id_dict)
    fig_obj.savefig(os.path.join(ddata_plots, f'Isomap_{x_key}_{n_neigh}.png'))

"""
Clustering using the metrics calcuated on the feature maps
"""
# Using MDS on precomputed distance maps

for x_key, x_vgg_metric in dict_vgg_metric.items():
    metric_name = x_key.split('_')[0]
    mds = MDS(dissimilarity='precomputed', n_init=10)
    X_transform = mds.fit_transform(x_vgg_metric)
    fig_obj = plot_embedding(X_transform, ssim_row_id_dict)
    fig_obj.savefig(os.path.join(ddata_plots, f'MDS_VGG_metric_{x_key}.png'))

"""
Plotting feature maps
"""

# STuff
ddest_vgg = '/data/seb/data/similarity_matrix/vgg_maps'
# Using MDS
chosen_indices = np.random.choice(range(n_points), 10)
for x_key, x_vgg in dict_vgg.items():
    for ii, i_img in enumerate(x_vgg[chosen_indices]):
        dest_file = os.path.join(ddest_vgg, f'vgg_{x_key}_{ii}.png')
        fig_title = f'label_{label_array[chosen_indices[ii]]}'
        fig_obj = hplotc.ListPlot(i_img, title=fig_title)
        fig_obj.figure.savefig(dest_file)
    hplotc.close_all()
