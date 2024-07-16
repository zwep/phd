from objective_configuration.inhomog_removal import PLOT_DIR, \
    IMG_VOLUNTEER, IMG_3T, \
    IMG_PATIENT, IMG_SYNTH, IMG_1p5T, \
    MASK_VOLUNTEER
import os
import collections
import helper.plot_class as hplotc
import numpy as np
import sklearn.cluster
from papers.inhomog_removal.rebuttal_work.FileGatherer import FileGather
from helper.clustering import GetEmbedding
import itertools

dir_list = {'volunteer': IMG_VOLUNTEER, '3T': IMG_3T, 'patient': IMG_PATIENT, 'synth': IMG_SYNTH, '1p5T': IMG_1p5T}

n_cluster = len(dir_list)
pca_list = []
img_list = []
key_list = []
n_img_dict = {}
for key, i_dir in dir_list.items():
    print(key)
    file_obj = FileGather(i_dir)
    img_array = np.array(file_obj.img_list)
    img_list.append(img_array)
    key_list.append(key)
    n_img_dict[key] = len(img_array)
    vgg_obj = GetEmbedding(data_array=img_array, feature_layer=3, model_name='dense')
    temp_pca = vgg_obj.get_pca_feature_array()
    pca_list.extend(temp_pca)


# Plot the histogram
# To demonstrate the little difference between the field strength in terms of pixel values
import matplotlib.pyplot as plt
fig, ax = plt.subplots(n_cluster, figsize=(15, 5))
for ii, i_img in enumerate(img_list):
    _ = ax[ii].hist(i_img[i_img > 0].ravel(), bins=256, range=(0, 1), density=True, label=key_list[ii])
    ax[ii].legend()



fig.savefig(os.path.join(PLOT_DIR, 'hist_img.png'))

"""
Below we tried to visualize the PCA'd features and cluster these. No luck


"""

# Get the minimum number of PCA components, otherwise comparisson is impossible due to different sizes
# I know this is annoying..
min_pca_comp = min([x.shape[0] for x in pca_list])
min_pca_comp = 2
truncated_pca_features = [x[:min_pca_comp].ravel() for x in pca_list]
truncated_pca_img = [x[:min_pca_comp] for x in pca_list]

plot_obj = hplotc.PlotCollage(content_list=[x[0] for x in truncated_pca_img], ddest=PLOT_DIR, n_display=6, plot_type='array')
plot_obj.plot_collage('_first')

plot_obj = hplotc.PlotCollage(content_list=[x[1] for x in truncated_pca_img], ddest=PLOT_DIR, n_display=6, plot_type='array')
plot_obj.plot_collage('_second')

# Here we can calculate the SSIM between the different PCA images
# This could show some clustering behavior already?
n_img = len(truncated_pca_img)
A = np.zeros((n_img, n_img))
for i_img in range(n_img):
    for j_img in range(i_img, n_img):
        x = truncated_pca_img[i_img][0]
        y = truncated_pca_img[j_img][0]
        from skimage.metrics import structural_similarity
        ssim = structural_similarity(x, y)
        A[i_img, j_img] = ssim
        A[j_img, i_img] = ssim

fig_obj = hplotc.ListPlot(A)
fig_obj.figure.savefig(os.path.join(PLOT_DIR, 'ssim_plot.png'))

# Here we use Kmeans to cluster the truncated PCA features
# This means that we have raveled every image
kmeans_obj = sklearn.cluster.KMeans(n_clusters=n_cluster).fit(truncated_pca_features)
kmean_labels = kmeans_obj.labels_
prev = 0
for k, v in n_img_dict.items():
    print(kmean_labels[prev: prev+v], collections.Counter(kmean_labels[prev: prev+v]).most_common())
    prev += v

# And here we check how close each cluster center is...
B = np.zeros((n_cluster, n_cluster))
for i_cluster in range(n_cluster):
    for j_cluster in range(i_cluster, n_cluster):
        x = kmeans_obj.cluster_centers_[i_cluster]
        y = kmeans_obj.cluster_centers_[j_cluster]
        l2_norm = np.sqrt(np.mean((x-y)**2))
        B[i_cluster, j_cluster] = l2_norm
        B[j_cluster, i_cluster] = l2_norm

fig_obj = hplotc.ListPlot(B, cbar=True)
fig_obj.figure.savefig(os.path.join(PLOT_DIR, 'cluster_distance.png'))
