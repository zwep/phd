import sklearn.cluster
import numpy as np
from itertools import islice
import helper.plot_class as hplotc
import pandas as pd
from objective_configuration.segment7T3T import DATASET_LIST, DATASET_SYNTH_LIST
import os

"""
Okay.. so we are going to cluster the image features we have gathered...
"""

dpng = '/home/bme001/20184098/visualization'

list_dataset = DATASET_LIST + DATASET_SYNTH_LIST
n_datasets = len(list_dataset)
n_cluster = n_datasets

# Load the feature data
A = pd.DataFrame()
size_dataset = []
for i_dataset in range(n_datasets):
    i_dataset_name = list_dataset[i_dataset]
    dpng_csv = os.path.join(dpng, i_dataset_name, 'image_feature.csv')
    B = pd.read_csv(dpng_csv, index_col=0)
    A = pd.concat([A, B])
    size_dataset.append(B.shape[0])

A_array = np.array(A)

# Fit Kmeans
kmeans_obj = sklearn.cluster.KMeans(n_clusters=n_cluster)
kmeans_obj.fit(A_array)
kmean_labels = kmeans_obj.labels_
# Chucnk kmeans labels back into the original datasets
it_labels = iter(kmean_labels)
sliced_labels = [list(islice(it_labels, 0, i)) for i in size_dataset]
# Print the average labels per dataset  (I know this is not complely OK)
for ii, i_group in enumerate(sliced_labels):
    i_dataset_name = list_dataset[ii]
    print(i_dataset_name, ' ' * (20 - len(i_dataset_name)), np.mean(i_group).round(2))

# Calculate the distance between the clusters to see something...?
B = np.zeros((n_cluster, n_cluster))
for i_cluster in range(n_cluster):
    for j_cluster in range(i_cluster, n_cluster):
        x = kmeans_obj.cluster_centers_[i_cluster]
        y = kmeans_obj.cluster_centers_[j_cluster]
        l1_norm = np.mean(np.abs(x-y))
        B[i_cluster, j_cluster] = l1_norm
        B[j_cluster, i_cluster] = l1_norm

fig_obj = hplotc.ListPlot(B, cbar=True)
fig_obj.figure.savefig(os.path.join(dpng, 'cluster_distance.png'))

# Now try to cluster with Bregman clustering
# https://www.jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf
from bregclus.models import BregmanHard
from bregclus.divergences import euclidean, squared_manhattan
import numpy as np

model_bregman = BregmanHard(n_clusters=n_datasets, divergence=squared_manhattan)
model_bregman.fit(A_array)
bregman_labels = model_bregman.predict(A_array)
it_labels = iter(bregman_labels)
sliced_labels = [list(islice(it_labels, 0, i)) for i in size_dataset]
# Print the average labels per dataset  (I know this is not complely OK)
for ii, i_group in enumerate(sliced_labels):
    i_dataset_name = list_dataset[ii]
    print(i_dataset_name, ' ' * (20 - len(i_dataset_name)), np.mean(i_group).round(2))

import matplotlib.pyplot as plt
# helper functions
def decision_region(clf, X):
    # Creating a grid in the data's range
    ran_x = X[:, 0].max() - X[:, 0].min()
    ran_y = X[:, 1].max() - X[:, 1].min()
    x = np.linspace(X[:, 0].min() - ran_x * 0.05, X[:, 0].max() + ran_x * 0.05, 256)
    y = np.linspace(X[:, 1].min() - ran_y * 0.05, X[:, 1].max() + ran_y * 0.05, 256)
    A, B = np.meshgrid(x, y)
    A_flat = A.reshape(-1, 1)
    B_flat = B.reshape(-1, 1)
    X_grid = np.hstack([A_flat, B_flat])
    # Defining a matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # compute predictions
    preds = clf.predict(X_grid)
    # Show the voronoi regions as an image
    ax.imshow(preds.reshape(A.shape), interpolation="bilinear", extent=(x.min(), x.max(), y.min(), y.max()),
              cmap="rainbow", aspect="auto", origin="lower", alpha=0.5)
    ax.axis("off")
    ax.set_xlabel("$x_1$");
    ax.set_ylabel("$x_2$")
    # Scatter plot of the original points
    ax.scatter(X[:, 0], X[:, 1], alpha=0.4, c="k")
    ax.set_xlim([X_grid[:, 0].min(), X_grid[:, 0].max()])
    ax.set_ylim([X_grid[:, 1].min(), X_grid[:, 1].max()])
    return fig, ax

fig, ax = decision_region(model_bregman, A_array)
fig.savefig(os.path.join(dpng, 'bregman_cluster.png'))