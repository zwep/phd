import small_project.homogeneity_measure.create_dummy_data as data_generator
import skimage.feature
import h5py
import scipy.io
import skimage.transform
import scipy.integrate
from skimage.util import img_as_ubyte
import skimage.feature
import helper.plot_class as hplotc
import matplotlib.pyplot as plt
import numpy as np
import helper.array_transf as harray
import helper.dummy_data as hdummy

"""
Example with austronaut data and B1 stuff
"""
A = skimage.data.astronaut()[:, :, 0]
res = hdummy.get_gaussian_blobs(*A.shape, n_c=2).sum(axis=0)
res = 2 * harray.scale_minmax(res) - 1

glcm_dist = [1, 2]
angles = [0, 2 * np.pi, np.pi // 2]

A_list = [A]
glcm_list = []
for biasf_factor in np.arange(0.01, 2, 0.01):
    simulated_biasf = np.sin(biasf_factor * res * np.pi) ** 3
    simulated_biasf = harray.scale_minmax(simulated_biasf)
    temp_A = A * simulated_biasf
    A_list.append(temp_A)
    i_glcm = skimage.feature.greycomatrix(temp_A.astype(int), distances=glcm_dist, angles=angles, levels=256, symmetric=True, normed=True)
    glcm_list.append(i_glcm)

hplotc.SlidingPlot(np.array(A_list))

homogeneity_list = [np.mean(skimage.feature.graycoprops(x, 'homogeneity')) for x in glcm_list]
correlation_list = [np.mean(skimage.feature.graycoprops(x, 'correlation')) for x in glcm_list]
energy_list = [np.mean(skimage.feature.graycoprops(x, 'energy')) for x in glcm_list]

fig, ax = plt.subplots()
ax.plot(homogeneity_list, label='homog')
# ax.plot(correlation_list, label='cor')
ax.plot(energy_list, label='energy')
plt.title(biasf_factor)
plt.legend()

# # Check again how a factor in/decreases affects the GLCM feauture..
temp_A = A[128:128+64, 128:128+64]
from skimage.util import img_as_ubyte
temp_A = img_as_ubyte(harray.scale_minmax(temp_A))
i_glcm = skimage.feature.greycomatrix(temp_A.astype(int), distances=glcm_dist, angles=angles, levels=int(temp_A.max())+1, symmetric=True, normed=True)
homog_value = skimage.feature.graycoprops(i_glcm, 'homogeneity').mean()
print(homog_value)

i_glcm = skimage.feature.greycomatrix((10+ temp_A).astype(int), distances=glcm_dist, angles=angles, levels=256+10, symmetric=True, normed=True)
homog_value = skimage.feature.graycoprops(i_glcm, 'homogeneity').mean()
print(homog_value)

fig, ax = plt.subplots(3)
ax[0].hist(A.ravel(), color='r', alpha=0.2)
ax[1].hist((2*A).ravel(), color='b', alpha=0.2)
ax[2].hist((20+A).ravel(), color='g', alpha=0.2)

"""
Another example!

"""

# Load Rho data (not registered to a specific b1+ map....
ddata_mrl = '/home/bugger/Documents/data/1.5T/prostate/4_MR/MRL/20201228_0004.h5'

with h5py.File(ddata_mrl, 'r') as f:
    temp = f['data']
    n_slice = temp.shape[0]
    A_rho = np.array(f['data'][n_slice//2])


A_rho = skimage.transform.resize(A_rho, (256, 256), preserve_range=True, anti_aliasing=False)
A_rho = harray.scale_minmax(A_rho)

"""
Load B1 data
"""
flavio_data = '/home/bugger/Documents/data/test_clinic_registration/flavio_data/M01.mat'

A = scipy.io.loadmat(flavio_data)
A_b1p = np.moveaxis(A['Model']['B1plus'][0][0], -1, 0)
A_b1m = np.moveaxis(A['Model']['B1minus'][0][0], -1, 0)
A_mask = A['Model']['Mask'][0][0]



data_obj = data_generator.DummyVaryingSignalData(rho=A_rho, b1p=A_b1p, b1m=A_b1m, mask=A_mask)
data_obj.selected_flip_angles_degree = np.arange(1, 180, 1)
varying_signal_map = data_obj.create_varying_signal_maps()

hplotc.SlidingPlot(varying_signal_map)
plt.figure()
plt.plot([np.std(x[A_mask == 1])/np.mean(x[A_mask == 1]) for x in varying_signal_map])

glcm_dist = [1, 2, 3, 4, 5, 6]
glcm_list = []
angles = [0, 2 * np.pi, np.pi // 2]
varying_signal_map = varying_signal_map * A_mask[None]
patch_width = min(varying_signal_map.shape[-2:]) // 4
mask_patches = harray.get_patches(A_mask, patch_shape=(patch_width, patch_width), stride=patch_width)
usefull_patch_indices = [i for i, x in enumerate(mask_patches) if np.sum(x) > (1/8 * patch_width * patch_width)]

rho_patches = harray.get_patches(A_rho * A_mask, patch_shape=(patch_width, patch_width), stride=patch_width)
rho_patches = rho_patches[usefull_patch_indices]

rho_glcm = []
for i_patch in rho_patches:
    i_patch = harray.scale_minmax(i_patch)
    i_patch = img_as_ubyte(i_patch)
    i_glcm = skimage.feature.greycomatrix(i_patch, distances=glcm_dist, angles=angles, levels=256,
                                          symmetric=True, normed=True)
    rho_glcm.append(i_glcm)

for temp_A in varying_signal_map:
    temp_patches = harray.get_patches(temp_A, patch_shape=(patch_width,patch_width), stride=patch_width)
    temp_patches = temp_patches[usefull_patch_indices]
    temp_list = []
    for ii, i_patch in enumerate(temp_patches):
        i_patch = harray.scale_minmax(i_patch)
        i_patch = img_as_ubyte(i_patch)
        i_glcm = skimage.feature.greycomatrix(i_patch, distances=glcm_dist, angles=angles, levels=256,
                                              symmetric=True, normed=True)
        temp_list.append(i_glcm)
    glcm_list.append(temp_list)

img_list = []
for patch_list in glcm_list:
    temp_list = []
    for i, ipatch in enumerate(patch_list):
        biasf_features = skimage.feature.graycoprops(ipatch, 'homogeneity')
        gt_features = skimage.feature.graycoprops(rho_glcm[i], 'homogeneity')
        rel_feature = np.mean((gt_features - biasf_features) / biasf_features)
        temp_list.append(rel_feature)
    img_list.append(temp_list)

energy_img_list = []
for patch_list in glcm_list:
    temp_list = []
    for i, ipatch in enumerate(patch_list):
        biasf_features = skimage.feature.graycoprops(ipatch, 'energy')
        gt_features = skimage.feature.graycoprops(rho_glcm[i], 'energy')
        rel_feature = np.mean((gt_features - biasf_features) / biasf_features)
        temp_list.append(rel_feature)
    energy_img_list.append(temp_list)

cor_img_list = []
for patch_list in glcm_list:
    temp_list = []
    for i, ipatch in enumerate(patch_list):
        biasf_features = skimage.feature.graycoprops(ipatch, 'correlation')
        gt_features = skimage.feature.graycoprops(rho_glcm[i], 'correlation')
        rel_feature = np.mean((gt_features - biasf_features) / biasf_features)
        temp_list.append(rel_feature)
    cor_img_list.append(temp_list)


rel_homogeneity_list = np.array(img_list).mean(axis=1)
rel_energy_list = np.array(energy_img_list).mean(axis=1)
rel_cor_list = np.array(cor_img_list).mean(axis=1)
plt.figure()
plt.plot(rel_homogeneity_list)
plt.plot(rel_cor_list)
plt.plot(rel_energy_list)

homogeneity_list = [np.mean([np.mean(skimage.feature.graycoprops(y, 'homogeneity')) for y in x]) for x in glcm_list]
rho_homogeneity_list = np.mean([np.mean(skimage.feature.graycoprops(y, 'homogeneity')) for y in rho_glcm])
# correlation_list = [np.mean([np.mean(skimage.feature.graycoprops(y, 'correlation')) for y in x]) for x in glcm_list]
energy_list = [np.mean([np.mean(skimage.feature.graycoprops(y, 'energy')) for y in x]) for x in glcm_list]
dissim_list = [np.mean([np.mean(skimage.feature.graycoprops(y, 'dissimilarity')) for y in x]) for x in glcm_list]

fig, ax = plt.subplots()
ax.plot(homogeneity_list, label='homog')
ax.plot(energy_list, label='energy')
#    ax.twinx().plot(dissim_list, label='energy')
plt.legend()
