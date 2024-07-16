import os
import numpy as np
import helper.array_transf as harray
import helper.plot_class as hplotc
import helper.misc as hmisc
import pydicom


ddir_uncor = '/home/bugger/Documents/data/7T/prostate/biasfield_correction/uncorrected'
ddir_cor_n4itk = '/home/bugger/Documents/data/7T/prostate/biasfield_correction/corrected_n4itk'
ddir_cor_biasf = '/home/bugger/Documents/data/7T/prostate/biasfield_correction/corrected_n4itk'
ddir_cor_homog = '/home/bugger/Documents/data/7T/prostate/biasfield_correction/corrected_homog'
ddest = '/home/bugger/Documents/data/7T/prostate/biasfield_correction'
file_list = os.listdir(ddir_uncor)
# sel_patient = file_list[16]
# file_path = os.path.join(ddir_cor, sel_patient)
# corrected_array = hmisc.load_array(file_path).T[:, ::-1, ::-1]

# hplotc.SlidingPlot(corrected_array, title=sel_patient)

from data_prep.dataset.prostate.daan_reesink.order_of_slices import slice_order_dict

# Now store the GIFs
for i_file in file_list:
    base_name = hmisc.get_base_name(i_file)
    sel_slice_order = slice_order_dict[base_name]
    # Doing the  homog corrected thing
    file_path = os.path.join(ddir_cor_homog, i_file)
    dest_file = os.path.join(ddest, 'homog_corrected_' + base_name + '.gif')
    corrected_array = hmisc.load_array(file_path).T[:, ::-1, ::-1]
    sorted_corrected_array = np.squeeze(corrected_array[sel_slice_order])
    n_slice = len(sorted_corrected_array)
    mask_slice = harray.get_treshold_label_mask(sorted_corrected_array[n_slice//2])
    # Get the crop coords to cut down the image in the y direction
    _, crop_coords = harray.get_center_transformation_coords(mask_slice)
    sorted_corrected_array = np.array([harray.apply_crop_axis(x, crop_coords=crop_coords, axis=0) for x in sorted_corrected_array])
    _, nx, ny = sorted_corrected_array.shape
    # conv_factor = nx/ny
    # hmisc.convert_image_to_gif(sorted_corrected_array, output_path=dest_file, n_card=n_slice, duration=8/n_slice, nx=256 * conv_factor, ny=256)
    # # Doing the  biasfg corrected thing
    # file_path = os.path.join(ddir_cor_biasf, i_file)
    # dest_file = os.path.join(ddest, 'biasf_corrected_' + base_name + '.gif')
    # uncorrected_array = hmisc.load_array(file_path).T[:, ::-1, ::-1]
    # sorted_uncorrected_array = np.squeeze(uncorrected_array[sel_slice_order])
    # sorted_uncorrected_array = np.array([harray.apply_crop_axis(x, crop_coords=crop_coords, axis=0) for x in sorted_uncorrected_array])
    # n_slice = len(sorted_uncorrected_array)
    # hmisc.convert_image_to_gif(sorted_uncorrected_array, output_path=dest_file, n_card=n_slice, duration=8 / n_slice, nx=256 * conv_factor, ny=256)
    # # Doing the uncorrected thing
    # file_path = os.path.join(ddir_uncor, i_file)
    # dest_file = os.path.join(ddest, 'uncorrected_' + base_name + '.gif')
    # uncorrected_array = hmisc.load_array(file_path).T[:, ::-1, ::-1]
    # sorted_uncorrected_array = np.squeeze(uncorrected_array[sel_slice_order])
    # cutoff_value = 150
    # sorted_uncorrected_array[sorted_uncorrected_array > cutoff_value] = cutoff_value
    # sorted_uncorrected_array = harray.scale_minmax(sorted_uncorrected_array)
    # sorted_uncorrected_array = np.array([harray.apply_crop_axis(x, crop_coords=crop_coords, axis=0) for x in sorted_uncorrected_array])
    # n_slice = len(sorted_uncorrected_array)
    # hmisc.convert_image_to_gif(sorted_uncorrected_array, output_path=dest_file, n_card=n_slice, duration=8/n_slice, nx=256 * conv_factor, ny=256)
    # Doing the n4itk thing
    file_path = os.path.join(ddir_cor_n4itk, i_file)
    dest_file = os.path.join(ddest, 'n4itk_corrected_' + base_name + '.gif')
    corrected_array = hmisc.load_array(file_path).T[:, ::-1, ::-1]
    sorted_corrected_array = np.squeeze(corrected_array[sel_slice_order])
    n_slice = len(sorted_corrected_array)
    mask_slice = harray.get_treshold_label_mask(sorted_corrected_array[n_slice//2])
    # Get the crop coords to cut down the image in the y direction
    _, crop_coords = harray.get_center_transformation_coords(mask_slice)
    sorted_corrected_array = np.array([harray.apply_crop_axis(x, crop_coords=crop_coords, axis=0) for x in sorted_corrected_array])
    _, nx, ny = sorted_corrected_array.shape
    conv_factor = nx/ny
    hmisc.convert_image_to_gif(sorted_corrected_array, output_path=dest_file, n_card=n_slice, duration=8/n_slice, nx=256 * conv_factor, ny=256)

d_1p5T_img = '/home/bugger/Documents/data/test_clinic_registration/registrated_h5/test/target_clean/M20_to_5_MR_20210329_0002_transversal.h5'
array_1p5T = hmisc.load_array(d_1p5T_img)
n_slice = array_1p5T.shape[0]
dest_file = os.path.join(ddest, 'low_field_strength.gif')
n_slice = len(array_1p5T)
mask_slice = harray.get_treshold_label_mask(array_1p5T[n_slice//2])
# Get the crop coords to cut down the image in the y direction
_, crop_coords = harray.get_center_transformation_coords(mask_slice)
sorted_cut_1p5T_img = np.array([harray.apply_crop(x, crop_coords=crop_coords) for x in array_1p5T[20:40]])
_, nx, ny = sorted_cut_1p5T_img.shape
hmisc.convert_image_to_gif(sorted_cut_1p5T_img, output_path=dest_file, n_card=n_slice, duration=8/n_slice, nx=nx//2, ny=ny//2)





# SOme work stuff...
import numpy as np
import matplotlib.pyplot as plt
import helper.plot_class as hplotc
ddata = '/home/bugger/Documents/data/7T/cardiac/example_3/v9_02052021_1124232.npy'
hplotc.ListPlot(np.load(ddata)[23])

# Some misc code evaluation stuff..?