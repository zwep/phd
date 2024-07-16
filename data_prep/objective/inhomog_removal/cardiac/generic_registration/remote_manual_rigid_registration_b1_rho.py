import nibabel
import time
import skimage.transform as sktransf
import skimage.transform
import scipy.ndimage
from skimage.util import img_as_ubyte, img_as_uint, img_as_int
import os
import numpy as np
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
import re
import argparse
import sys
from objective_configuration.segment7T3T import DCMR7T
from nnunet.paths import nnUNet_raw_data


parser = argparse.ArgumentParser()
parser.add_argument('-vendor', type=str)
parser.add_argument('-dataset', type=str)
parser.add_argument('-debug', type=bool, default=False)
parser.add_argument('-overwrite', type=bool, default=False)
p_args = parser.parse_args()

vendor = p_args.vendor
dataset = p_args.dataset
debug_status = p_args.debug
overwrite = p_args.overwrite

"""
Here we register B1 to any type of cardiac data..

It is a really ugly script. Should have made more classes and methods for this. But this monster just grew and kept growing

"""

# dataset = 'acdc'
# vendor = 'A'
# debug_status = True
# overwrite = False

ddata_b1 = os.path.join(DCMR7T, 'b1_distr/sa')
ddata_b1_rho = os.path.join(ddata_b1, 'rho')
ddata_b1_segmentation_mask = os.path.join(ddata_b1, 'segm_mask')
ddata_b1_minus = os.path.join(ddata_b1, 'b1_minus')
ddata_b1_plus = os.path.join(ddata_b1, 'b1_plus')
ddata_b1_mask = os.path.join(ddata_b1, 'mask')

b1_file_names = sorted(os.listdir(ddata_b1_rho))
n_b1 = len(b1_file_names)

train_perc = 0.80
test_perc = 0.20

n_train_b1 = int(n_b1 * train_perc)
n_test_b1 = int(n_b1 * test_perc)
list_b1_train = b1_file_names[:n_train_b1]
list_b1_test = b1_file_names[n_train_b1:]


if dataset == 'mm1':
    if vendor == 'A':
        ddata_cardiac = os.path.join(nnUNet_raw_data, 'Task501_MM1_A')
    elif vendor == 'B':
        ddata_cardiac = os.path.join(nnUNet_raw_data, 'Task502_MM1_B')
    elif vendor == 'test':
        ddata_cardiac = '/data/seb/data/test_full_fov_mm1_a'
        ddest = f'/data/cmr7t3t/biasfield_sa_mm1_{vendor}_test'
    else:
        print('stuff')
        sys.exit()
    ddest = os.path.join(DCMR7T, f'biasfield_sa_mm1_{vendor}')
elif dataset == 'acdc':
    # ddata_cardiac = '/data/cmr7t3t/acdc/acdc_processed'
    ddata_cardiac = os.path.join(nnUNet_raw_data, 'Task511_ACDC')
    ddest = os.path.join(DCMR7T, 'biasfield_sa_acdc')
else:
    print('No known dataset ', dataset)
    sys.exit()

if not os.path.isdir(ddest):
    os.makedirs(ddest)

ddest_examples = os.path.expanduser('~')
# Need to add the Train/Text appendix name still
temp_ddata_cardiac_img = os.path.join(ddata_cardiac, 'images')
temp_ddata_cardiac_label = os.path.join(ddata_cardiac, 'labels')


# Determine how to split everything into train/test/validation...
hmisc.create_datagen_dir(ddest, type_list=('test', 'train'), data_list=('input', 'target', 'mask', 'target_segmentation', 'target_clean'))

# Now all the training and such has been written down
for data_type in ['train', 'test']:
    print(f'Starting to process ={data_type}= data')
    if data_type == 'train':
        ddata_cardiac_img = temp_ddata_cardiac_img + 'Tr'
        ddata_cardiac_label = temp_ddata_cardiac_label + 'Tr'
        selected_b1_list = list_b1_train
    elif data_type == 'test':
        ddata_cardiac_img = temp_ddata_cardiac_img + 'Ts'
        ddata_cardiac_label = temp_ddata_cardiac_label + 'Ts'
        selected_b1_list = list_b1_test
    else:
        selected_cardiac_list = []
        selected_b1_list = []
        print('unknown data type')
        sys.exit()

    selected_cardiac_list = sorted(os.listdir(ddata_cardiac_label))
    print('Example of B1 files')
    print(selected_b1_list)
    print('Example of cardiac files')
    print(selected_cardiac_list[0:3], " ... ", selected_cardiac_list[-3:])
    if debug_status:
        b1_item = int(input(f'Index of b1 (i in 0, {len(selected_b1_list)})'))
        cardiac_item = int(input(f'Index of cardiac (i in 0, {len(selected_cardiac_list)})'))
        selected_b1_list = selected_b1_list[b1_item:b1_item + 1]
        selected_cardiac_list = selected_cardiac_list[cardiac_item: cardiac_item + 1]
#
    for i, sel_b1_file in enumerate(selected_b1_list):
        print(f"Processing {sel_b1_file}, {i} / {len(selected_b1_list)}", end='\r')
        b1_file_name, _ = os.path.splitext(sel_b1_file)
        for sel_cardiac_file in selected_cardiac_list:
            if debug_status:
                print('Using b1 file ', sel_b1_file)
                print('Using cardiac file ', sel_cardiac_file)
            t0 = time.time()
            cardiac_file_name = hmisc.get_base_name(sel_cardiac_file)
            cardiac_file_ext = hmisc.get_ext(sel_cardiac_file)
            file_name = f'{b1_file_name}_to_{cardiac_file_name}'

            # We are going to use the segmentation masks to align the images..
            b1_segmentation_mask_file = os.path.join(ddata_b1_segmentation_mask, sel_b1_file)
            cardiac_segmentation_mask_file = os.path.join(ddata_cardiac_label, sel_cardiac_file)
            # Load the arrays
            b1_segmentation_mask_array = hmisc.load_array(b1_segmentation_mask_file)
            temp_cardiac_segmentation_mask_array = hmisc.load_array(cardiac_segmentation_mask_file)
            if temp_cardiac_segmentation_mask_array.ndim == 4:
                print(temp_cardiac_segmentation_mask_array.shape)
                n_loc = temp_cardiac_segmentation_mask_array.shape[-2]
                n_phase = temp_cardiac_segmentation_mask_array.shape[-1]
                sel_phase = n_phase//2
                temp_cardiac_segmentation_mask_array = temp_cardiac_segmentation_mask_array[:, :, :, sel_phase]
                # Dont take all locations, since the outer most are a bit corrupted and unclear...
                if n_loc > 3:
                    range_locations = np.arange(n_loc//2-2, n_loc//2+2)
                else:
                    range_locations = range(n_loc)
            elif temp_cardiac_segmentation_mask_array.ndim == 3:
                n_loc = temp_cardiac_segmentation_mask_array.shape[-1]
                # Dont take all locations, since the outer most are a bit corrupted and unclear...
                if n_loc > 3:
                    range_locations = np.arange(n_loc//2-2, n_loc//2+2)
                else:
                    range_locations = range(n_loc)
            else:
                temp_cardiac_segmentation_mask_array = temp_cardiac_segmentation_mask_array[:, :, None]
                n_loc = 1
                range_locations = range(n_loc)

            # I dont want all locations.. only the middle ones..

            for sel_loc in range_locations:
                # Create the storage file names... They are stored as npy
                new_file_name = file_name
                if n_loc > 1:
                    new_file_name = file_name + f"_loc_{sel_loc}"
                ddest_b1_minus = os.path.join(ddest, data_type, 'input', new_file_name)
                ddest_b1_plus = os.path.join(ddest, data_type, 'target', new_file_name)
                ddest_mask = os.path.join(ddest, data_type, 'mask', new_file_name)
                ddest_target_segmentation = os.path.join(ddest, data_type, 'target_segmentation', new_file_name)
                ddest_target_clean = os.path.join(ddest, data_type, 'target_clean', new_file_name)
                # If one already exists then just continue
                if (overwrite is False) and (debug_status is False):
                    if os.path.isfile(ddest_b1_minus + '.npy'):
                        print('Skipping file ')
                        print(f'vendor_{vendor}_{data_type} ', sel_b1_file, sel_cardiac_file)
                        continue

                cardiac_segmentation_mask_array = temp_cardiac_segmentation_mask_array[:, :, sel_loc]
                if debug_status:
                    print('Current loaded size')
                    print('B1 mask array ', b1_segmentation_mask_array.shape)
                    print('Cardiac segm array ', cardiac_segmentation_mask_array.shape)
                    fig_obj = hplotc.ListPlot([[b1_segmentation_mask_array, cardiac_segmentation_mask_array]])
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_loaded_segmentation_mask.png'),
                                           bbox_inches='tight', pad_inches=0.0)
                # Registration stuff...
                """Determine the scaling size by comparing the difference in x- and y- size of both cardiac segmentations"""
                # Determine size of MM1 segmentation
                _, crop_coords_cardiac = harray.get_center_transformation_coords(cardiac_segmentation_mask_array)
                distance_segmentation_cardiac = np.array([crop_coords_cardiac[1] - crop_coords_cardiac[0], crop_coords_cardiac[3] - crop_coords_cardiac[2]])
                # Determine size of B1 segmentation
                _, crop_coords_b1 = harray.get_center_transformation_coords(b1_segmentation_mask_array)
                distance_segmentation_b1 = np.array([crop_coords_b1[1] - crop_coords_b1[0], crop_coords_b1[3] - crop_coords_b1[2]])
                # The obtained scaling factor:
                scale_factor = max(distance_segmentation_cardiac / distance_segmentation_b1)
                b1_segmentation_mask_scaled = skimage.transform.rescale(b1_segmentation_mask_array, scale=scale_factor,
                                                                              preserve_range=True, anti_aliasing=False)
                affine_coords_b1_scaled, _ = harray.get_center_transformation_coords(
                    (b1_segmentation_mask_scaled > 0).astype(int))
                midcoords_b1_scaled = np.abs(np.array(b1_segmentation_mask_scaled.shape) // 2 - affine_coords_b1_scaled)
                """ Now that the same 'size' is obtained, we want to cut-out the appropiate part of the b1 file
                This is done by finding the distance of the center of hart in the M&M1 file to the boundary of the file
                This size is then used, with respect to the center, to cut out the appropiate part of the b1 file"""
                if debug_status:
                    # Lets check what we are going now...
                    fig_obj = hplotc.ListPlot(cardiac_segmentation_mask_array)
                    nx, ny = cardiac_segmentation_mask_array.shape
                    fig_obj.ax_list[0].hlines(crop_coords_cardiac[3], 0, ny - 1)
                    fig_obj.ax_list[0].hlines(crop_coords_cardiac[2], 0, ny - 1)
                    fig_obj.ax_list[0].vlines(crop_coords_cardiac[1], 0, nx - 1)
                    fig_obj.ax_list[0].vlines(crop_coords_cardiac[0], 0, nx - 1)
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_initial_cardiac_box.png'),
                                           bbox_inches='tight', pad_inches=0.0)
                    fig_obj = hplotc.ListPlot(b1_segmentation_mask_array)
                    nx, ny = b1_segmentation_mask_array.shape
                    fig_obj.ax_list[0].hlines(crop_coords_b1[3], 0, ny-1)
                    fig_obj.ax_list[0].hlines(crop_coords_b1[2], 0, ny-1)
                    fig_obj.ax_list[0].vlines(crop_coords_b1[1], 0, nx-1)
                    fig_obj.ax_list[0].vlines(crop_coords_b1[0], 0, nx-1)
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_initial_b1_box.png'),
                                           bbox_inches='tight', pad_inches=0.0)
                # But first we want to know how to crop the newly sized image to be somewhat aligned with the cardiac dataset.
                # So. Lets determine what the distance is from the center of the segmentationentation to the edge..
                affine_coords_cardiac, _ = harray.get_center_transformation_coords((cardiac_segmentation_mask_array > 0).astype(int))
                min_coords_cardiac = np.abs(np.array(affine_coords_cardiac) - np.array(cardiac_segmentation_mask_array.shape) // 2)
                max_coords_cardiac = np.array(cardiac_segmentation_mask_array.shape) - np.abs(np.array(affine_coords_cardiac) - np.array(cardiac_segmentation_mask_array.shape) // 2)
                #
                # Pad more zeros to it.. so that we can always make the appropriate cut
                b1_segmentation_mask_padded = np.pad(b1_segmentation_mask_scaled, ((100, 100), (100, 100)))
                affine_coords_b1_padded, _ = harray.get_center_transformation_coords((b1_segmentation_mask_padded > 0).astype(int))
                midcoords_b1_padded = np.abs(np.array(b1_segmentation_mask_padded.shape) // 2 - affine_coords_b1_padded)
                x_0_b1, y_0_b1 = midcoords_b1_padded - min_coords_cardiac
                x_1_b1, y_1_b1 = midcoords_b1_padded + max_coords_cardiac
                # Now that we have the proper coordinates.. lets apply them to the mask and find the optimial rotation
                # Lets check if we can optimize a certain rotation...
                b1_segmentation_mask_cropped = b1_segmentation_mask_padded[y_0_b1:y_1_b1, x_0_b1:x_1_b1]
                # This is the middle of the segmentation mask now..
                p0 = np.array([midcoords_b1_padded[0] - x_0_b1, midcoords_b1_padded[1] - y_0_b1])
                # b1_segmentation_mask_rotated, opt_degree = harray.optimize_rotation(b1_segmentation_mask_cropped, cardiac_segmentation_mask_array, p=p0)
                if debug_status:
                    # b1_segmentation_mask_rotated
                    plot_array = [b1_segmentation_mask_array, b1_segmentation_mask_scaled, b1_segmentation_mask_padded,
                                  b1_segmentation_mask_cropped, cardiac_segmentation_mask_array]
                    fig_obj = hplotc.ListPlot(plot_array, col_row=(2, 3), title=f'scaling factor used {scale_factor}')
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_operations_to_b1_mask.png'),
                                           bbox_inches='tight', pad_inches=0.0)
                    hplotc.close_all()
                """ Now we have the (or a(n)) scaling, padding and rotation scheme that can be applied to the B1 distributions"""
                # Load the B1+
                b1_plus_file = os.path.join(ddata_b1_plus, sel_b1_file)
                b1_plus_array = np.load(b1_plus_file)
                # Load the B1-
                b1_minus_file = os.path.join(ddata_b1_minus, sel_b1_file)
                b1_minus_array = np.load(b1_minus_file)
                # Create a body mask from the img array
                cardiac_file = os.path.join(ddata_cardiac_img, cardiac_file_name + "_0000" + cardiac_file_ext)
                cardiac_array = hmisc.load_array(cardiac_file)
                if cardiac_array.ndim == 4:
                    cardiac_array = cardiac_array[:, :, :, sel_phase]

                if cardiac_array.ndim == 2:
                    cardiac_array = cardiac_array[:, :, None]
                cardiac_array = cardiac_array[:, :, sel_loc]
                mask_array_cardiac = harray.get_treshold_label_mask(cardiac_array, class_treshold=0.001, treshold_value=0.01)
                # --> Apply it to the B1+ array...
                b1_plus_resize = np.array([harray.rescale_complex_array(x, scale=scale_factor,
                                                                           preserve_range=True,
                                                                           anti_aliasing=False) for x in b1_plus_array])
                b1_plus_padded = np.pad(b1_plus_resize, ((0, 0), (100, 100), (100, 100)))
                b1_plus_crop = b1_plus_padded[:, y_0_b1:y_1_b1, x_0_b1:x_1_b1]

                if debug_status:
                    # Here we also display other things...

                    b1_for_plotting = harray.scale_minmax(np.abs(b1_plus_array).sum(axis=0)) + 0.5*(b1_segmentation_mask_array)
                    b1_resize_for_plotting = harray.scale_minmax(np.abs(b1_plus_resize).sum(axis=0)) + 0.5*(b1_segmentation_mask_scaled)
                    b1_padded_for_plotting = harray.scale_minmax(np.abs(b1_plus_padded).sum(axis=0)) + 0.5*(b1_segmentation_mask_padded)
                    b1_crop_for_plotting = harray.scale_minmax(np.abs(b1_plus_crop).sum(axis=0)) + 0.5*(b1_segmentation_mask_cropped)

                    fig_obj = hplotc.ListPlot([[b1_for_plotting, cardiac_array + 0.5*(cardiac_segmentation_mask_array>0).astype(int)],
                                               [b1_resize_for_plotting, cardiac_array],
                                               [b1_padded_for_plotting, cardiac_array],
                                               [b1_crop_for_plotting, cardiac_array]], wspace=0.00, hspace=0.15,
                                              cmap=[['viridis', 'gray'], ['viridis', 'gray'], ['viridis', 'gray'], ['viridis', 'gray']])
                    # Add lines to axis 0
                    nx, ny = b1_segmentation_mask_array.shape
                    fig_obj.ax_list[0].hlines(crop_coords_b1[3], 0, ny - 1)
                    fig_obj.ax_list[0].hlines(crop_coords_b1[2], 0, ny - 1)
                    fig_obj.ax_list[0].vlines(crop_coords_b1[1], 0, nx - 1)
                    fig_obj.ax_list[0].vlines(crop_coords_b1[0], 0, nx - 1)
                    nx, ny = cardiac_segmentation_mask_array.shape
                    fig_obj.ax_list[1].hlines(crop_coords_cardiac[3], 0, ny - 1)
                    fig_obj.ax_list[1].hlines(crop_coords_cardiac[2], 0, ny - 1)
                    fig_obj.ax_list[1].vlines(crop_coords_cardiac[1], 0, nx - 1)
                    fig_obj.ax_list[1].vlines(crop_coords_cardiac[0], 0, nx - 1)

                    # midcoords_b1_padded - min_coords_cardiac[0]
                    # x_1_b1, y_1_b1 = midcoords_b1_padded + max_coords_cardiac
                    fig_obj.ax_list[4].scatter(*midcoords_b1_padded, c='r')
                    # Sanity check
                    fig_obj.ax_list[4].scatter(*(midcoords_b1_padded - min_coords_cardiac), c='r')
                    fig_obj.ax_list[4].scatter(*(midcoords_b1_padded + max_coords_cardiac), c='r')

                    x_low = (midcoords_b1_padded - min_coords_cardiac)[0]
                    y_low = (midcoords_b1_padded - min_coords_cardiac)[1]
                    x_high = (midcoords_b1_padded + max_coords_cardiac)[0]
                    y_high = (midcoords_b1_padded + max_coords_cardiac)[1]
                    fig_obj.ax_list[4].vlines(x_high, y_low, y_high, color='r')
                    fig_obj.ax_list[4].vlines(x_low, y_low, y_high, color='r')
                    fig_obj.ax_list[4].hlines(y_high, x_low, x_high, color='r')
                    fig_obj.ax_list[4].hlines(y_low, x_low, x_high, color='r')
                    fig_obj.ax_list[5].scatter(*np.array(min_coords_cardiac), color='r')

                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_overview_operations.png'), bbox_inches='tight')
                    # b1_segmentation_mask_rotated
                    # b1_segmentation_mask_rotated
                    plot_array = [np.abs(b1_plus_array).sum(axis=0), np.abs(b1_plus_resize).sum(axis=0),
                                               np.abs(b1_plus_padded).sum(axis=0), np.abs(b1_plus_crop).sum(axis=0)]
                    fig_obj = hplotc.ListPlot(plot_array, col_row=(2, 2), cmap='viridis')
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_operations_to_b1_plus.png'),
                                           bbox_inches='tight', pad_inches=0.0)
                    hplotc.close_all()
                print(f'vendor_{vendor}_{data_type}_File status ', sel_b1_file, sel_cardiac_file)
                print('Size of b1 plus crop', b1_plus_crop.shape)
                if any(np.array(b1_plus_crop.shape) == 0):
                    print('Wrong shape.. continue ')
                    continue
                b1_plus_crop = harray.scale_minmax(b1_plus_crop, is_complex=True, axis=(-2,-1))
                b1_plus_crop = b1_plus_crop / np.abs(b1_plus_crop).max()
                b1_plus_stacked = np.stack([img_as_int(b1_plus_crop.real), img_as_int(b1_plus_crop.imag)])
                #
                b1_minus_resize = np.array([harray.rescale_complex_array(x, scale=scale_factor,
                                                                         preserve_range=True,
                                                                         anti_aliasing=False) for x in b1_minus_array])
                b1_minus_padded = np.pad(b1_minus_resize, ((0, 0), (100, 100), (100, 100)))
                b1_minus_crop = b1_minus_padded[:, y_0_b1:y_1_b1, x_0_b1:x_1_b1]
                if debug_status:
                    # # #
                    # b1_segmentation_mask_rotated
                    plot_array = [np.abs(b1_minus_array).sum(axis=0), np.abs(b1_minus_resize).sum(axis=0),
                                  np.abs(b1_minus_padded).sum(axis=0), np.abs(b1_minus_crop).sum(axis=0)]
                    print("shape of operations to b1 minus")
                    print([x.shape for x in plot_array])
                    fig_obj = hplotc.ListPlot(plot_array, col_row=(2, 2), cmap='viridis')
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_operations_to_b1_minus.png'))
                    hplotc.close_all()
                b1_minus_crop = harray.scale_minmax(b1_minus_crop, is_complex=True, axis=(-2,-1))
                b1_minus_crop = b1_minus_crop / np.abs(b1_minus_crop).max()
                b1_minus_stack = np.stack([img_as_int(b1_minus_crop.real), img_as_int(b1_minus_crop.imag)])

                if debug_status:
                    print('Shape of B1-', b1_minus_stack.shape)
                    print('Shape of B1+', b1_plus_stacked.shape)
                    print('Cardiac array', cardiac_array.shape)
                    print('Cardiac segm', cardiac_segmentation_mask_array.shape)
                    print(mask_array_cardiac.shape)

                # Now do all the storage stuff..
                # Dont save it when we are debugging
                if not debug_status:
                    np.save(ddest_b1_minus, b1_minus_stack.astype(int))
                    np.save(ddest_b1_plus, b1_plus_stacked.astype(int))
                    np.save(ddest_target_clean, img_as_ubyte(cardiac_array))
                    np.save(ddest_target_segmentation, cardiac_segmentation_mask_array.astype(np.int8))
                    np.save(ddest_mask, mask_array_cardiac.astype(bool))

                # Store the final and starting image to demonstrat the inhomogeneity added.
                if debug_status:
                    import tooling.shimming.b1shimming_single as mb1_single
                    shim_mask = (cardiac_segmentation_mask_array > 0).astype(int)
                    b1_plus_crop = harray.scale_minmax(b1_plus_crop, is_complex=True, axis=(-2, -1))
                    b1_plus_crop = b1_plus_crop * np.exp(-1j * np.angle(b1_plus_crop[0]))
                    shimming_obj = mb1_single.ShimmingProcedure(b1_plus_crop, shim_mask, relative_phase=True, str_objective='signal_ge')
                    x_opt, final_value = shimming_obj.find_optimum()
                    b1_plus_crop_shimmed = harray.apply_shim(b1_plus_crop, cpx_shim=x_opt)
                    #
                    b1_minus_crop = harray.scale_minmax(b1_minus_crop, is_complex=True, axis=(-2, -1))
                    b1_minus_crop = b1_minus_crop * np.exp(-1j * np.angle(b1_minus_crop[0]))
                    shimming_obj = mb1_single.ShimmingProcedure(b1_minus_crop, shim_mask, relative_phase=True,
                                                                str_objective='signal_ge')
                    x_opt, final_value = shimming_obj.find_optimum()
                    b1_minus_crop_shimmed = harray.apply_shim(b1_minus_crop, cpx_shim=x_opt)
                    #
                    biasfield = np.abs(b1_minus_crop).sum(axis=0) * np.abs(b1_plus_crop_shimmed)
                    final_biasf_image_example = cardiac_array * biasfield
                    # # Plot b1 plus array
                    max_b1_plus = 0.8 * np.max(np.abs(b1_plus_padded))
                    fig_obj = hplotc.ListPlot(b1_plus_padded, cbar=True, augm='np.abs', vmin=(0, max_b1_plus))
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_b1_plus_abs.png'))
                    fig_obj = hplotc.ListPlot(b1_plus_padded, cbar=True, augm='np.angle')
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_b1_plus_angle.png'))
                    # # Plot b1 minus array
                    max_b1_min = 0.8*np.max(np.abs(b1_minus_padded))
                    fig_obj = hplotc.ListPlot(b1_minus_padded, cbar=True, augm='np.abs', vmin=(0, max_b1_min))
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_b1_minus_abs.png'))
                    fig_obj = hplotc.ListPlot(b1_minus_padded, cbar=True, augm='np.angle')
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_b1_minus_angle.png'))
                    # # Plot shimming vs abs sum
                    fig_obj = hplotc.ListPlot([[np.abs(b1_plus_crop.sum(axis=0)), np.abs(b1_plus_crop_shimmed), shim_mask]], cbar=True)
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_b1_plus_shimmed.png'))
                    # # Also for b1minus
                    fig_obj = hplotc.ListPlot([[np.abs(b1_minus_crop.sum(axis=0)), np.abs(b1_minus_crop_shimmed), shim_mask]], cbar=True)
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_b1_minus_shimmed.png'))
                    # Plot the original and bias field corrupted image
                    fig_obj = hplotc.ListPlot([[cardiac_array, final_biasf_image_example]])
                    fig_obj.figure.savefig(os.path.join(ddest_examples, f'vendor_{vendor}_{data_type}_final_example.png'))
                    hplotc.close_all()
