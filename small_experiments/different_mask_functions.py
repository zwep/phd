
"""
Try out some masking algorithms on a single prostate_mri_mrl image..
"""
import skimage.morphology as skmorphology
import skimage.segmentation as sksegmentation
import scipy.ndimage

import pydicom
import os
import numpy as np
import scipy.io
import helper.plot_class as hplotc
import helper.array_transf as harray

# This one is not split into test/train/val yet...
prostate_dir = "/home/bugger/Documents/data/1.5T/prostate"
cine_dir = '/media/bugger/MyBook/data/7T_scan/cardiac_dicom/2021_03_07/V9_17994/DICOM_5_1_Cine1SliceR2_tra/DICOM'

# Lets just test the class first...
prostate_file_list = [x for x in os.listdir(prostate_dir) if x.endswith('.dcm')]
cine_file_list = [x for x in os.listdir(cine_dir)]
n_prostate_files = len(prostate_file_list)
n_cine_files = len(cine_file_list)

i_counter = 0
sel_file = prostate_file_list[i_counter]
prostate_file = os.path.join(prostate_dir, sel_file)

sel_file = cine_file_list[i_counter]
cine_file = os.path.join(cine_dir, sel_file)

prostate_array = pydicom.read_file(prostate_file).pixel_array
sel_time = 0
cine_array = pydicom.read_file(cine_file).pixel_array[sel_time]

array_to_mask = cine_array
# Based on Otsu tresholding..
otsu_mask = harray.get_otsu_mask(array_to_mask)
# Based on simple tresholding and labeling of continous areas
label_mask = harray.get_treshold_label_mask(array_to_mask)
# Based on simple convolution...
smooth_mask = harray.get_smoothed_mask(array_to_mask)
# Felzens zwalb
felzenszwalb_mask = sksegmentation.felzenszwalb(array_to_mask)
# Chan vsese mask
chan_vese_mask = sksegmentation.morphological_chan_vese(array_to_mask, 100)
chan_vese_mask = scipy.ndimage.binary_fill_holes(1 - chan_vese_mask)
# Tresholded mask
treshold_mask = array_to_mask > np.mean(array_to_mask)
treshold_mask = scipy.ndimage.binary_fill_holes(treshold_mask)
# Using dilation
skel, distance = skmorphology.medial_axis(array_to_mask > np.mean(array_to_mask), return_distance=True)
dilated = skmorphology.dilation(skel * distance, selem=skmorphology.selem.disk(4))
dilated = scipy.ndimage.binary_fill_holes(dilated)

hplotc.ListPlot([[otsu_mask, smooth_mask, felzenszwalb_mask, chan_vese_mask, treshold_mask, label_mask, dilated]], augm='np.abs')
