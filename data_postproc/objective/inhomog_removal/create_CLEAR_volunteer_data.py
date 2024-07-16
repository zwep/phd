"""
Here we are going to collect all the model outputs that give the undistrubed/restored as result immediately

"""

import matplotlib.pyplot as plt
import helper.plot_class as hplotc
import numpy as np
import torch
import helper.misc as hmisc
import helper.array_transf as harray
import objective.inhomog_removal.executor_inhomog_removal as executor
import os
import glob

"""
Prostate directories
"""

main_data_path = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/'
measured_path = os.path.join(main_data_path, 't2w')
body_mask_path = os.path.join(main_data_path, 'body_mask')
prostate_mask_path = os.path.join(main_data_path, 'prostate_mask')
muscle_mask_path = os.path.join(main_data_path, 'muscle_mask')
subcutaneous_fat_mask_path = os.path.join(main_data_path, 'subcutaneous_fat_mask')


"""
Creating lists of files for masks
"""

body_mask_file_list = sorted([os.path.join(body_mask_path, x) for x in os.listdir(body_mask_path)])
prostate_mask_file_list = sorted([os.path.join(prostate_mask_path, x) for x in os.listdir(prostate_mask_path)])
muscle_mask_file_list = sorted([os.path.join(muscle_mask_path, x) for x in os.listdir(muscle_mask_path)])
subcutaneous_fat_mask_file_list = sorted([os.path.join(subcutaneous_fat_mask_path, x) for x in os.listdir(subcutaneous_fat_mask_path)])
# These nr of files can be more than the amount of files there are...
# SO just check against ANY mask dir, to match the files. Then sort
# This makes sure that file 0 is "the same" over all the lists
file_list = sorted([os.path.join(measured_path, x) for x in os.listdir(measured_path) if x in os.listdir(subcutaneous_fat_mask_path)])
dest_multi_pred_path = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/prediction_multi'
dest_pred_path = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/prediction'

for i in range(len(file_list)):
    print('Processing', i)
    # for i in [18, 17, 16, 13, 12, 11, 6, 5, 4, 0]:
    load_file = file_list[i]
    file_name, _ = os.path.splitext(os.path.basename(load_file))
    body_mask_file = body_mask_file_list[i]
    prostate_mask_file = prostate_mask_file_list[i]
    muscle_mask_file = muscle_mask_file_list[i]
    subcutaneous_fat_mask_file = subcutaneous_fat_mask_file_list[i]

    input_cpx = np.load(load_file)
    body_mask_array = np.load(body_mask_file)

    # # Make the clear image...
    # Sometimes..
    hplotc.ListPlot(input_cpx, augm='np.angle')
    sensitivity_map = input_cpx / np.sum(np.abs(input_cpx), axis=0)
    hplotc.ListPlot(sensitivity_map, augm='np.abs')
    sensitivity_map_summed = np.sum(np.abs(sensitivity_map) ** 3, axis=0)
    hplotc.ListPlot(sensitivity_map_summed, augm='np.abs')
    scaled_image = np.einsum("cxy, cxy -> xy", input_cpx, sensitivity_map)
    # nx = 50
    # ny = 20
    ### Validate the calculation of X \cdot S  (both are 3d matrices)
    # Validate how we calculate the einsum. Make sure that we understand it
    # (input_cpx[:, nx, ny] * sensitivity_map[:, nx, ny]).sum()
    # scaled_image[nx, ny]

    # Unpack the X * S_i a little more to chekc calculations and simplifications
    # (input_cpx[:, nx, ny] * input_cpx[:, nx, ny] * (1 / np.abs(input_cpx[:, nx, ny]).sum(axis=0))).sum(axis=0)
    # Unpack a bit more
    # (1 / np.abs(input_cpx[:, nx, ny]).sum(axis=0)) * (input_cpx[:, nx, ny] * input_cpx[:, nx, ny]).sum(axis=0)
    # nominator = (1 / np.abs(input_cpx[:, nx, ny]).sum(axis=0)) * (input_cpx[:, nx, ny] * input_cpx[:, nx, ny]).sum(axis=0)
    # nominator = (input_cpx[:, nx, ny] * input_cpx[:, nx, ny]).sum(axis=0)
    # nominator = (np.abs(input_cpx[:, nx, ny]).sum(axis=0)) * (input_cpx[:, nx, ny] * input_cpx[:, nx, ny]).sum(axis=0)

    # Now check the demoninator S (squared abs sum of the sensitivity maps)
    # sensitivity_map_summed[nx, ny]
    # ((abs(input_cpx[:, nx, ny] * (1 / np.abs(input_cpx[:, nx, ny]).sum(axis=0))))**2).sum(axis=0)
    # (1 / np.abs(input_cpx[:, nx, ny]).sum(axis=0)) ** 2 * (abs(input_cpx[:, nx, ny])**2).sum(axis=0)
    # (1 / np.abs(input_cpx[:, nx, ny]).sum(axis=0)) ** 2 * (input_cpx[:, nx, ny] * input_cpx[:, nx, ny].conjugate()).sum(axis=0)

    # denominator = (1 / np.abs(input_cpx[:, nx, ny]).sum(axis=0)) ** 2 * (input_cpx[:, nx, ny] * input_cpx[:, nx, ny].conjugate()).sum(axis=0)
    # denominator = (input_cpx[:, nx, ny] * input_cpx[:, nx, ny].conjugate()).sum(axis=0)
    # denominator = (input_cpx[:, nx, ny] * input_cpx[:, nx, ny].conjugate()).sum(axis=0)

    clear_image = np.abs(scaled_image / sensitivity_map_summed)
    nominator = np.einsum("cxy, cxy -> xy", input_cpx, input_cpx)
    scale_factor = np.sum(np.abs(input_cpx), axis=0)
    denominator = np.einsum("cxy, cxy -> xy", input_cpx, input_cpx.conjugate())
    clear_image_easier = np.abs(scale_factor * nominator / denominator)

    hplotc.ListPlot([clear_image, clear_image_easier])

    clear_image = harray.scale_minmax(clear_image) * body_mask_array
    hplotc.ListPlot([clear_image, np.abs(input_cpx.sum(axis=0))], title='power 2')
    from skimage.util import img_as_ubyte, img_as_int, img_as_uint
    import SimpleITK as sitk
    single_target_dir_patient = os.path.join(dest_pred_path, file_name)
    multi_target_dir_patient = os.path.join(dest_multi_pred_path, file_name)
    # Store them in both locations...
    single_clear_path = os.path.join(single_target_dir_patient, f'uncorrected_clear.dcm')
    multi_clear_path = os.path.join(single_target_dir_patient, f'uncorrected_clear.dcm')
    # WRite corrected file
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(single_clear_path)
    res = img_as_uint(clear_image)
    writer.Execute(sitk.GetImageFromArray(res))
    # WRite corrected file
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(multi_clear_path)
    res = img_as_uint(clear_image)
    writer.Execute(sitk.GetImageFromArray(res))