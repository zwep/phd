"""
Lets see how easy it is to get all the cardiac cropped images..
"""

import os
import numpy as np
import helper.array_transf as harray
import helper.misc as hmisc

dataset = None

if dataset in ['7T', '7t']:
    # Location of the ground truth labels
    ddata_img = '/data/cmr7t3t/cmr7t/Image_ED_ES'
    ddata_label = '/data/cmr7t3t/cmr7t/Label_ED_ES'
elif dataset in ['3T', '3t', 'acdc']:
    # Location of the ground truth labels
    ddata_img = '/data/cmr7t3t/acdc/acdc_processed/ImageTest'
    ddata_label = '/data/cmr7t3t/acdc/acdc_processed/LabelTest'
elif dataset in ['1p5T', '1p5t', 'mm1']:
    # Location of the ground truth labels
    # This location now has the proper labels.. i.e. # class_interpretation = {'3': 'RV', '2': 'MYO', '1': 'LV'}
    ddata_img = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task501_MM1_A/imagesTr'
    ddata_label = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task501_MM1_A/labelsTr'
elif dataset in ['biasf']:
    ddata_img = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task567_Biasfield_ACDC/imagesTr'
    ddata_label = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task567_Biasfield_ACDC/labelsTr'
elif dataset in ['synth']:
    ddata_img = '/data/cmr7t3t/results/ACDC_220121/seven2three_acdc_cut_NCE4_GAN2_np128_fe_211208/test_100/niftis/cmr3t2cmr7t'
    ddata_label = '/data/cmr7t3t/results/ACDC_220121/seven2three_acdc_cut_NCE4_GAN2_np128_fe_211208/test_100/niftis/cmr3t_label'


# Now I got all the paths...
# Now I need to get all the data..
# And get a 'distance'
"""
The choices are........

d(X,Y) - Perceptual Style Loss
d(X,Y) - PCA on VGG embedding, then Eucledian distance
d(X,Y) - SSIM - Similarity
d(X,Y) - SSIM - Contrast
d(X,Y) - WD

"""