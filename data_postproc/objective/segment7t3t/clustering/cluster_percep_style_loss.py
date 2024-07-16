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
    fig, ax = plt.subplots(dpi=300)
    for ii, (ikey, iamount) in enumerate(row_label_dict.items()):
        x0 += prev_amount
        x1 += iamount
        ax.scatter(embedding[x0:x1, 0], embedding[x0:x1, 1], color=cmap(ii), label=ikey, s=16)
        prev_amount = iamount
    ax.legend()
    return fig


def plot_pred_label(embedding, label, fig=None):
    # Assuming the fig has an axes in it...
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]
    label_array = np.array(label)
    unique_labels = set(label)
    n_labels = len(unique_labels)
    cmap = matplotlib.cm.get_cmap('Reds', lut=n_labels+1)
    for i_label in list(set(label)):
        index_label = label_array == i_label
        sel_embedding = embedding[index_label]
        ax.scatter(sel_embedding[:, 0], sel_embedding[:, 1], color=cmap(i_label), label=i_label+1, s=4)
    ax.legend()
    return fig


ddata = '/data/seb/data/similarity_matrix'
ddata_plot = '/data/seb/data/similarity_matrix/cluster_img'
# prefix = ''
prefix = 'cropped_'
distance_style = np.load(os.path.join(ddata, f'{prefix}distance_style.npy'))
distance_perception = np.load(os.path.join(ddata, f'{prefix}distance_perception.npy'))

similarity_style = distance_style + distance_style.T
similarity_style[np.diag_indices(similarity_style.shape[0])] /= 2

similarity_percep = distance_perception + distance_perception.T
similarity_percep[np.diag_indices(similarity_percep.shape[0])] /= 2

fig_obj = hplotc.ListPlot([similarity_style, similarity_percep], subtitle=[['style'], ['percep']])
fig_obj.figure.savefig(os.path.join(ddata_plot, f'{prefix}style_percep_distance_mat.png'))

# Load the id behind the data points...
with open(os.path.join(ddata, 'key_length_string.txt'), 'r') as f:
    temp_read = f.read()

temp_parsed = [x.split(":") for x in temp_read.split("\n")]

ssim_row_id_dict = OrderedDict()
for i_item in temp_parsed:
    if len(i_item) > 1:
        x_key, x_value = i_item
        ssim_row_id_dict.update({x_key: int(x_value)})

dict_ssim = {'style': similarity_style,
             'percep': similarity_percep}

# Using UMAP
for x_key, x_sim in dict_ssim.items():
    # Distances: 0 is close...
    # Similarity: 1 is close...
    # What does UMAP need?
    # Welll..... this is a very nice answer
    # https://stats.stackexchange.com/questions/2717/clustering-with-a-distance-matrix
    # And this gives also a nice overview
    # https://en.wikipedia.org/wiki/Multidimensional_scaling
    # THis gives a great overivew of TSNE
    # https://distill.pub/2016/misread-tsne/
    # similarity_matrix = np.exp(-distance_matrix ** 2 / (2. * delta ** 2))
    reducer = umap.UMAP(metric='precomputed', init='random')
    x_embed = reducer.fit_transform(x_sim)
    fig_obj = plot_embedding(x_embed, ssim_row_id_dict)
    fig_obj.savefig(os.path.join(ddata_plot, f'{prefix}UMAP_percep_style_{x_key}.png'))

# Using MDS
for x_key, x_sim in dict_ssim.items():
    # Try clustering based on Multidimensional Scaling...
    mds = MDS(dissimilarity='precomputed', n_init=10)
    X_transform = mds.fit_transform(x_sim)
    fig_obj = plot_embedding(X_transform, ssim_row_id_dict)
    fig_obj.savefig(os.path.join(ddata_plot, f'{prefix}MDS_percep_style_{x_key}.png'))


# Using Isomap
for x_key, x_sim in dict_ssim.items():
    # Try clustering based on Multidimensional Scaling...
    ismap_obj = Isomap(metric='precomputed')
    X_transform = ismap_obj.fit_transform(x_sim)
    fig_obj = plot_embedding(X_transform, ssim_row_id_dict)
    fig_obj.savefig(os.path.join(ddata_plot, f'{prefix}Isomap_percep_style_{x_key}.png'))


# Try clustering based on Multidimensional Scaling...
for alpha in np.arange(0, 1, 0.1):
    alpha = np.round(alpha, 2)
    ismap_obj = Isomap(metric='precomputed')
    X_transform = ismap_obj.fit_transform(similarity_percep * alpha + (1-alpha) * similarity_style)
    fig_obj = plot_embedding(X_transform, ssim_row_id_dict)
    fig_obj.savefig(os.path.join(ddata_plot, f'{prefix}Isomap_comb_percep_style_{alpha}.png'))


# What if we cluster in high dimensions and use embedding of other things..
import sklearn.cluster
dbscan_obj = sklearn.cluster.dbscan(X=similarity_percep, metric='precomputed')
n_clusters = len(ssim_row_id_dict)
kmeans_obj = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(similarity_percep)
# Plot some embdding with the KMeans labels...
ismap_obj = Isomap(metric='precomputed')
X_transform = ismap_obj.fit_transform(similarity_percep)


fig = plot_embedding(X_transform, ssim_row_id_dict)
fig = plot_pred_label(X_transform, kmeans_obj.labels_, fig=fig)
fig.savefig(os.path.join(ddata, 'test.png'))