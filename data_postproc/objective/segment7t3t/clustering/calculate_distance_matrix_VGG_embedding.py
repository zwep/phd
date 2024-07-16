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

"""
Here we calculate the SSIM/WSS/CONTRAST for the VGG Embeddings..

"""

def get_metric(X):
    # I Would like to get these metrics (SSIM, WSS, Contrast) on the feature maps as well
    n_img = X.shape[0]
    ssim_matrix = np.zeros((n_img, n_img))
    contrast_matrix = np.zeros((n_img, n_img))
    wss_matrix = np.zeros((n_img, n_img))
    for i_img in range(n_img):
        print(f"{i_img} / {n_img}", end='\r')
        x_i = X[i_img]
        x_i = harray.scale_minmax(x_i)
        x_i_mean = x_i.mean()
        x_hist_i, _ = np.histogram(x_i.ravel(), bins=256, range=(0, 1))
        for j_img in range(i_img, n_img):
            x_j = X[j_img]
            x_j = harray.scale_minmax(x_j)
            x_j_mean = x_j.mean()
            x_j = x_j * (x_i_mean / x_j_mean)
            x_hist_j, _ = np.histogram(x_j.ravel(), bins=256, range=(0, 1))
            wss_value = scipy.stats.wasserstein_distance(x_hist_i, x_hist_j)
            ssim_value = structural_similarity(x_i, x_j)
            contrast_value = hmetric.get_contrast_ssim(x_i, x_j)
            ssim_matrix[i_img, j_img] = ssim_value
            contrast_matrix[i_img, j_img] = contrast_value
            wss_matrix[i_img, j_img] = wss_value
    return ssim_matrix, contrast_matrix, wss_matrix


ddata = '/data/seb/data/similarity_matrix'
# Preparing for cropped prefix...
# prefix = ''
prefix = 'cropped_'

dict_vgg = {}
vgg_0 = np.load(os.path.join(ddata, f'{prefix}pca_feature_array_vgg_0.npy'))
dict_vgg['vgg_0'] = vgg_0
vgg_1 = np.load(os.path.join(ddata, f'{prefix}pca_feature_array_vgg_1.npy'))
dict_vgg['vgg_1'] = vgg_1
vgg_2 = np.load(os.path.join(ddata, f'{prefix}pca_feature_array_vgg_2.npy'))
dict_vgg['vgg_2'] = vgg_2
vgg_3 = np.load(os.path.join(ddata, f'{prefix}pca_feature_array_vgg_3.npy'))
dict_vgg['vgg_3'] = vgg_3


# Calculate metrics on the feature maps..
for x_key, x_vgg in dict_vgg.items():
    ssim_matrix, contrast_matrix, wss_matrix = get_metric(x_vgg)
    ddata = '/data/seb/data/similarity_matrix'
    np.save(os.path.join(ddata, f'{prefix}ssim_matrix_{x_key}.npy'), ssim_matrix)
    np.save(os.path.join(ddata, f'{prefix}contrast_matrix_{x_key}.npy'), contrast_matrix)
    np.save(os.path.join(ddata, f'{prefix}wss_matrix_{x_key}.npy'), wss_matrix)
