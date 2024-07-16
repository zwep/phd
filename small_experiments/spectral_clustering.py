import matplotlib.pyplot as plt
import matplotlib
import helper.dummy_data as hdata
import helper.plot_class as hplotc
import sklearn.cluster
import numpy as np

"""
Experiment with dummy data
"""

n_blobs = 6
A = hdata.get_gaussian_blobs(25, 25, n_blobs)
B = (A > 0.5).sum(axis=0)
# Create a distance and similarity graph
B_coords = np.argwhere(B)
n_points = B_coords.shape[0]
cpx_points = (B_coords[:, 0] + 1j * B_coords[:, 1]).reshape((n_points, 1))
distance_matrix = np.abs(cpx_points.T - cpx_points)
delta = 20
similarity_matrix = np.exp(-distance_matrix ** 2 / (2. * delta ** 2))

"""
Here we go.. for the 2D data.. which is easier since we can plot stuff 
"""

n_points = similarity_matrix.shape[0]
# Clustering with sklearn cluster
cluster_obj = sklearn.cluster.SpectralClustering(n_clusters=n_blobs, affinity='precomputed', n_neighbors=5)
spectral_cluster_labels = cluster_obj.fit_predict(similarity_matrix)
coords_labels = np.concatenate([spectral_cluster_labels.reshape(n_points, 1), B_coords], axis=-1)

# Nu clustering doen met eigen algoritme..
diag_elements = np.sum(similarity_matrix, axis=1)
D = np.diag(diag_elements)
D_sqrt_inv = np.diag(1 / np.sqrt(diag_elements))
# k Means algorithm..
L_sym = np.eye(n_points) - (D_sqrt_inv) @ similarity_matrix @ (D_sqrt_inv)
L_rw = np.eye(n_points) - np.linalg.inv(D) @ similarity_matrix
eigvalue, eigvector = np.linalg.eig(L_sym)
test = np.sum(~np.isclose(eigvalue.real, 1, atol=0.1))
U = eigvector[:, :test]
kmeans_obj = sklearn.cluster.KMeans(n_clusters=n_blobs)
kmeans_obj.fit(U.real)
kmean_labels = kmeans_obj.labels_
kmean_coords_labels = np.concatenate([kmean_labels.reshape(n_points, 1), B_coords], axis=-1)
hplotc.ListPlot(kmeans_obj.cluster_centers_)
plt.scatter(kmeans_obj.cluster_centers_[:, 1], kmeans_obj.cluster_centers_[:, 2])

# Clustering based on ...
fig, ax = plt.subplots()
cmap = matplotlib.cm.get_cmap('plasma', lut=n_blobs)
for i_point in coords_labels:
    ilabel, ix, iy = i_point
    ax.scatter(ix, iy, color=cmap(ilabel))

# Clustering based on self created Laplacian
fig, ax = plt.subplots()
cmap = matplotlib.cm.get_cmap('plasma', lut=n_blobs)
for i_point in kmean_coords_labels:
    ilabel, ix, iy = i_point
    ax.scatter(ix, iy, color=cmap(ilabel))

