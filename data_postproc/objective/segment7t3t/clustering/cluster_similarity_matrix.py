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


ddata = '/data/seb/data/similarity_matrix'
ddata_plot = '/data/seb/data/similarity_matrix/cluster_img'
# prefix = ''
prefix = 'cropped_'
ssim_matrix = np.load(os.path.join(ddata, f'{prefix}ssim_matrix.npy'))
contrast_matrix = np.load(os.path.join(ddata, f'{prefix}contrast_matrix.npy'))
wss_matrix = np.load(os.path.join(ddata, f'{prefix}wss_matrix.npy'))


similarity_ssim = ssim_matrix + ssim_matrix.T
similarity_ssim[np.diag_indices(similarity_ssim.shape[0])] /= 2
similarity_contrast = contrast_matrix + contrast_matrix.T
similarity_contrast[np.diag_indices(similarity_contrast.shape[0])] /= 2
similarity_wasserstein = wss_matrix + wss_matrix.T


fig_obj = hplotc.ListPlot([similarity_ssim, similarity_contrast, similarity_wasserstein], subtitle=[['ssim'], ['contrast'], ['wss']])
fig_obj.figure.savefig(os.path.join(ddata_plot, 'ssim_contrast_wss_matrix.png'))

# Load the id behind the data points...
with open(os.path.join(ddata, 'key_length_string.txt'), 'r') as f:
    temp_read = f.read()

temp_parsed = [x.split(":") for x in temp_read.split("\n")]

ssim_row_id_dict = OrderedDict()
for i_item in temp_parsed:
    if len(i_item) > 1:
        x_key, x_value = i_item
        ssim_row_id_dict.update({x_key: int(x_value)})

dict_ssim = {'contrast': similarity_contrast,
             'wasserstein': similarity_wasserstein,
             'ssim': similarity_ssim}

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
    fig_obj.savefig(os.path.join(ddata_plot, f'{prefix}UMAP_{x_key}.png'))

# Using MDS
for x_key, x_sim in dict_ssim.items():
    # Try clustering based on Multidimensional Scaling...

    mds = MDS(dissimilarity='precomputed', n_init=10)
    if x_key == 'ssim':
        x_sim = (1-x_sim) / 2
    X_transform = mds.fit_transform(x_sim)
    fig_obj = plot_embedding(X_transform, ssim_row_id_dict)
    fig_obj.savefig(os.path.join(ddata_plot, f'{prefix}MDS_{x_key}.png'))


# Using Isomap
for i_neigh in range(2, 20, 5):
    for x_key, x_sim in dict_ssim.items():
        # Try clustering based on Multidimensional Scaling...

        ismap_obj = Isomap(metric='precomputed', n_neighbors=i_neigh)
        if x_key == 'ssim':
            x_sim = (1-x_sim) / 2
        X_transform = ismap_obj.fit_transform(x_sim)
        fig_obj = plot_embedding(X_transform, ssim_row_id_dict)
        fig_obj.savefig(os.path.join(ddata_plot, f'{prefix}Isomap_{x_key}_{i_neigh}.png'))


#
#
# # Nu clustering doen met eigen algoritme..
# # Select a similarity matrix...
# similarity = similarity_ssim
# n_points = similarity.shape[0]
# diag_elements = np.sum(similarity, axis=1)
# D = np.diag(diag_elements)
# D_sqrt_inv = np.diag(1 / np.sqrt(diag_elements))
# # k Means algorithm..
# L_sym = np.eye(n_points) - (D_sqrt_inv) @ similarity @ (D_sqrt_inv)
# L_rw = np.eye(n_points) - np.linalg.inv(D) @ similarity
# # Take the L_sym as Laplacian matrix..
# eigvalue, eigvector = np.linalg.eig(L_sym)
# test = np.sum(~np.isclose(eigvalue.real, 1, atol=0.1))
# U = eigvector[:, :test]
#
# # Use Kmeans for clustering
# import sklearn.cluster
# kmeans_obj = sklearn.cluster.KMeans(n_clusters=n_clusters)
# kmeans_obj.fit(U.real)
# kmean_labels = kmeans_obj.labels_
# # plot_pred_labels(kmean_labels, rowiddict=ssim_row_id_dict)
#
# for i in range(n_clusters):
#     for j in range(i, n_clusters):
#         a = kmeans_obj.cluster_centers_[i, :]
#         b = kmeans_obj.cluster_centers_[j, :]
#         dist = np.linalg.norm(a - b)
#         print(i, j, dist)
#
# # Nu met Spectral Clustering..
# cluster_obj = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_neighbors=5)
# spectral_cluster_labels = cluster_obj.fit_predict(similarity)
# coords_labels = np.concatenate([spectral_cluster_labels.reshape(n_points, 1), B_coords], axis=-1)



