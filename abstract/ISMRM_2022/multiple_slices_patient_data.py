from PIL import Image
import scipy.optimize

from skimage.util import img_as_ubyte, img_as_uint, img_as_int
import pydicom
import objective.inhomog_removal.recall_inhomog_removal as inhom_recall
import helper.misc as hmisc
import os
import numpy as np
import helper.array_transf as harray
import helper.plot_class as hplotc

"""
Here we create multiple slice print....
"""

dest_dir = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Seb_abstract_ISMRM'
recal_obj = inhom_recall.RecallInhomog()

# Load a test data set... let's say the complex one.
main_data_path = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Seb_pred/'

# Number of slices we need...
delta_slice = 5

predicted_image = []
subject_name = []
for ipatient in os.listdir(main_data_path):
    patient_dir = os.path.join(main_data_path, ipatient)

    uncorrected = os.path.join(patient_dir, 'uncorrected.dcm')
    biasfield_pred = hmisc.get_latest_file(patient_dir, file_filter='biasfield')

    array_biasfield_pred = pydicom.read_file(biasfield_pred).pixel_array
    array_uncorrected = pydicom.read_file(uncorrected).pixel_array

    n_slice = array_uncorrected.shape[0]
    sel_slice = n_slice // 2
    if '020' in ipatient:
        sel_slice = 11
    elif '016' in ipatient:
        sel_slice = 11
    elif '013' in ipatient:
        sel_slice = 11

    stuff = {'uncorrected': array_uncorrected[sel_slice-delta_slice:sel_slice+delta_slice],
             'single_biasf': array_biasfield_pred[sel_slice-delta_slice:sel_slice+delta_slice]}
    predicted_image.append(stuff)
    subject_name.append(ipatient)

# Find proper scaling based on a single patch...
sel_index = -1
sel_index = subject_name.index('7TMRI017')
for sel_index in range(len(predicted_image)):
    sel_index += 1
    sel_labels = list(predicted_image[sel_index].keys())
    sel_labels = np.repeat(sel_labels, 2*delta_slice)
    sel_images = list(predicted_image[sel_index].values())
    sel_images = np.reshape(sel_images, (-1, 1024, 1024))
    current_name = subject_name[sel_index]
    sel_images = [harray.scale_minmax(x) for x in sel_images]

    patch_obj = hplotc.PatchVisualizer(sel_images[0], patch_width=256)
    # patch_obj = hplotc.PatchVisualizer()
    # pos_x, pos_y = np.array(sel_images[0].shape) // 2
    # patch_obj.point_list.append((pos_x, pos_y))
    # reference_image = sel_images[0]
    # patch_img = patch_obj.get_patch(pos_y, pos_x, img=reference_image)
    # hplotc.ListPlot([reference_image, patch_img])

    all_patches = []
    for i_img in sel_images:
        temp = []
        for ix, iy in patch_obj.point_list:
            temp_patch = patch_obj.get_patch(iy, ix, img=i_img)
            temp.append(temp_patch)
        all_patches.append(temp)

    reference_patches = all_patches[0]
    remaining_patches = all_patches[1:]

    n_patches = len(reference_patches)
    n_models = len(remaining_patches)
    temp_scale_list = []
    for i in range(n_models):
        temp_scale = []
        for ii in range(n_patches):
            res = scipy.optimize.differential_evolution(hmisc.patch_min_fun_ssim, strategy='randtobest1exp',
                                                        bounds=[(0, 20)], args=(remaining_patches[i][ii], reference_patches[ii]), x0=1)

            temp_scale.append(res.x[0])
        temp_scale_list.append(temp_scale)

    mean_scaling = np.mean(temp_scale_list, axis=(-1))

    reference_image = sel_images[0]
    remaining_images = sel_images[1:]
    res_images = remaining_images * (np.array(mean_scaling)[:, None, None])

    patch_shape = tuple(np.array(reference_image.shape) // 10)
    stride = patch_shape[0] // 2
    vmax = harray.get_proper_scaled_v2(reference_image, patch_shape, stride)

    res_images = [reference_image] + list(res_images)
    for ii in range(len(res_images)):
        res_images[ii][res_images[ii] > vmax] = vmax

    res_images = [harray.scale_minmax(x) for x in res_images]
    all_patches = [x[0] for x in all_patches]
    all_patches[0] = harray.scale_minmax(all_patches[0])
    # hplotc.close_all()
    res_images = [harray.scale_minmax(x) for x in res_images]

    plot_obj = None
    # plot_obj = hplotc.ListPlot(np.array(res_images)[None], subtitle=[sel_labels], start_square_level=4, ax_off=True,  cbar=True, wspace=0, hspace=0, title=f'{current_name}')
    plot_obj = hplotc.ListPlot(np.array(res_images)[None], start_square_level=2, ax_off=True, cbar=True)
    print(current_name)

    # import pyperclip
    # pyperclip.copy(current_name)

    if plot_obj is not None:
        # Rescale the images based on the manually scaled images...
        for iindex in range(len(res_images)):
            cmin, cmax = plot_obj.ax_imshow_list[iindex].get_clim()
            # res_images[iindex] = (res_images[iindex] - cmin) / (cmax - cmin)
            res_images[iindex][res_images[iindex] > cmax] = cmax
            res_images[iindex][res_images[iindex] < cmin] = cmin
            res_images[iindex] = harray.scale_minmax(res_images[iindex])

    target_dir = os.path.join(dest_dir, current_name)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    counter = 0
    for x_image, y_label in zip(res_images, sel_labels):
        print(counter, y_label)
        target_file = os.path.join(target_dir, y_label + f'_{str(counter).zfill(2)}')
        img_obj = Image.fromarray(img_as_uint(x_image))
        img_obj.save(target_file + '.png')
        counter += 1
        counter = counter % (2*delta_slice)

## This was to check how the contrast changes...
import scipy.ndimage

hplotc.SlidingPlot(np.array(res_images))
# Small test to check how the contrast behaves in the centre..
orig1 = res_images[6]
bisaf1 = res_images[16]
mask_obj = hplotc.MaskCreator(orig1)
import helper.array_transf as harray

orig1_crop, mask_crop = harray.get_crop(orig1, mask_obj.mask)
biasf1_crop, mask_crop = harray.get_crop(bisaf1, mask_obj.mask)

hplotc.ListPlot([orig1_crop, biasf1_crop])
