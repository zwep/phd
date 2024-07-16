"""
Lets try some stuff with clustering....

TSNE - t-SNE
DBSCAN - ..?
SOM - Self Organizing Map
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tooling.tsne_plot as ttsne_plot
import skimage.transform as sktransform
import helper.misc as hmisc
import helper.array_transf as harray
import numpy as np
import os

# Get the images from all the directories
ddata_7T = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task999_7T/imagesTs'
ddata_3T = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task511_ACDC/imagesTr'
ddata_1p5T = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task433_MM1_A/imagesTs'
# Exclude files iwth the name 'orig' in it
ddata_biasfield_7T = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task612_ACDC_Biasfield_ACDC/imagesTr'
ddata_synthetic_7T = '/data/cmr7t3t/results/ACDC_220121/seven2three_acdc_cut_NCE4_GAN2_np128_fe_211208/test_100/niftis/cmr3t2cmr7t'

n_limit = 1000
data_array_7T = hmisc.load_dir(ddata_7T, n_limit=n_limit)
label_array_7T = ['7T'] * len(data_array_7T)

data_array_3T = hmisc.load_dir(ddata_3T, n_limit=n_limit)
label_array_3T = ['3T'] * len(data_array_3T)

data_array_1p5T = hmisc.load_dir(ddata_1p5T, n_limit=n_limit)
label_array_1p5T = ['1p5T'] * len(data_array_1p5T)

data_array_7T_biasfield = hmisc.load_dir(ddata_biasfield_7T, n_limit=n_limit, filter_string='orig')
label_array_7T_biasfield = ['7T_biasfield'] * len(data_array_7T_biasfield)

data_array_7T_synth = hmisc.load_dir(ddata_synthetic_7T, n_limit=n_limit)
label_array_synth = ['7T_synth'] * len(data_array_7T_synth)

data_array = np.concatenate([data_array_7T, data_array_3T, data_array_1p5T, data_array_7T_biasfield, data_array_7T_synth])
label_array = label_array_7T + label_array_3T + label_array_1p5T + label_array_7T_biasfield + label_array_synth

print("Size of data array ", data_array.shape)
for i_model in ['vgg', 'dense']:
    for i_layer in [2, 3]:
        tsne_plot_obj = ttsne_plot.TSNEPlot(data_array, label_array)
        fig_obj = tsne_plot_obj.plot_features_tsne()
        fig_obj.savefig(f'/data/seb/tsne_plot_{i_model}_{str(i_layer).zfill(2)}.png')