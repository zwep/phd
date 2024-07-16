
"""
Here we actually register the images/masks together.

We have the following masks
    - b1 minuis
    - b1 plus
    - rho

We need to transform everything onto the real magnitude images to perserve detail

"""

import SimpleITK as sitk
import numpy as np
import os
import re
import getpass
import json

import skimage.transform as sktrans
import matplotlib.pyplot as plt
import copy
import sys

if getpass.getuser() == 'bugger':
    local_system = True
    base_dir = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/'
    project_path = '/'

else:
    import matplotlib as mpl
    mpl.use('Agg')  # Hopefully this makes sure that we can plot/save stuff
    local_system = False
    base_dir = '/data/seb/semireal/cardiac_simulation_rxtx/'
    project_path = '/home/seb/code/pytorch_in_mri'

sys.path.append(project_path)

import helper.plot_fun as hplotf
import helper.array_transf as harray
import helper.misc as hmisc


# This is about the masks....
def register_mask(x_orig, y_orig, param_vector, debug=False, mask=False):
    """We map x_orig to y_orig.."""
    elastix_obj = sitk.ElastixImageFilter()
    if not debug:
        elastix_obj.LogToConsoleOff()

    min_scale = min([y / x for x, y in zip(x_orig.shape[-2:], y_orig.shape[-2:])])

    x_scaled = sktrans.rescale(x_orig, min_scale, order=3, preserve_range=True)
    # This is to make the bouandries nice and smooth again
    x_scaled[x_scaled < 0.9] = 0
    x_scaled = np.ceil(x_scaled).astype(int)

    if debug:
        print('min scale ', min_scale)
        print('x input shape: ', x_orig.shape, ' --> ', x_scaled.shape)
        plt.hist(x_scaled.ravel(), bins=30, density=True, log=True,
                 label='x scaled', alpha=0.5)
        print('y input shape ', y_orig.shape)
        plt.hist(y_orig.ravel(), bins=30, density=True, log=True,
                 label='y orig', alpha=0.5)
        plt.legend()

        fig, ax = plt.subplots(2)
        ax[0].imshow(x_scaled)
        ax[0].set_title('x_scaled')
        ax[1].imshow(x_orig)
        ax[1].set_title('x_original')

    # Convert to Image object
    moving_image = sitk.GetImageFromArray(x_scaled.astype(int))
    fixed_image = sitk.GetImageFromArray(y_orig.astype(int))

    elastix_obj.SetMovingImage(moving_image)
    elastix_obj.SetFixedImage(fixed_image)

    if mask:
        print('Using a mask')
        moving_image_int = sitk.Cast(moving_image, sitk.sitkUInt8)
        fixed_image_int = sitk.Cast(fixed_image, sitk.sitkUInt8)
        elastix_obj.SetMovingMask(moving_image_int)
        elastix_obj.SetFixedMask(fixed_image_int)

    elastix_obj.SetParameterMap(param_vector)

    transformed_image = elastix_obj.Execute()
    transformed_array = sitk.GetArrayFromImage(transformed_image)

    if debug:
        print('Output of transformation ', transformed_array.shape)
        hplotf.plot_3d_list([transformed_array, x_scaled, y_orig],
                            subtitle=[['transformed'], ['moving'], ['fixed']])

    transform_mapping = elastix_obj.GetTransformParameterMap()
    return transform_mapping, min_scale


def validate_mask_mapping(x_orig, transform_mapping, min_scale, y_orig=None, plot=False):
    x_scaled = sktrans.rescale(x_orig, min_scale, order=3, preserve_range=True)
    # This is to make the bouandries nice and smooth again
    x_scaled[x_scaled < 0.9] = 0
    x_scaled = np.ceil(x_scaled).astype(int)

    x_image = sitk.GetImageFromArray(x_scaled)
    validate_image = sitk.Transformix(x_image, transform_mapping)
    validate_array = sitk.GetArrayFromImage(validate_image)

    if plot:
        if y_orig is not None:
            fig_handle = hplotf.plot_3d_list([validate_array, x_orig, y_orig],
                                subtitle=[['transformed'], ['moving'], ['target']])
        else:
            fig_handle = hplotf.plot_3d_list([validate_array, x_orig])
    else:
        fig_handle = -1

    return validate_array, fig_handle


def apply_transform(x_orig, min_scale, transform_mapping, mask):
    if x_orig.ndim == 2:
        x_orig = x_orig[np.newaxis]
    else:
        print('Input dimension is ', x_orig.ndim)

    if 'complex' in str(x_orig.dtype):
        print('Warning! Input is complex. Continuing with real values')

    x_scaled = np.array([sktrans.rescale(x, min_scale, order=3, preserve_range=True) for x in x_orig])

    temp = []
    for x in x_scaled:
        x_image = sitk.GetImageFromArray(x)
        x_transformed = sitk.Transformix(x_image, transform_mapping)
        x_array = sitk.GetArrayFromImage(x_transformed)

        if mask is not None:
            # Correct with mask
            x_array = x_array * mask
            mask_int = ((1 - mask) == 1).astype(int)
            input_array_close = np.isclose(x_array, 0).astype(int)
            input_array_outside = (input_array_close * mask_int).astype(bool)
            x_array[input_array_outside] = 0

        temp.append(x_array)

    temp_stacked = np.stack(temp, axis=0)
    return temp_stacked


plot_intermediate_results = False

# Define directories for retrieval and storage
dir_b1_minus = os.path.join(base_dir, 'non_registered/b1minus/sel_filtered_aligned')
dir_b1_plus = os.path.join(base_dir, 'non_registered/b1plus')
dir_rho = os.path.join(base_dir, 'non_registered/rho')
dest_dir = os.path.join(base_dir, 'registered')

hmisc.create_datagen_dir(dest_dir, type_list=['train', 'test', 'validation'], data_list=['input', 'mask', 'target', 'target_clean'])

rho_files = os.listdir(dir_rho)
b1_minus_files = os.listdir(dir_b1_minus)
b1_plus_files = os.listdir(dir_b1_plus)

# # # Define parameter maps for registration
# Get the default parameter maps
affine_map = sitk.GetDefaultParameterMap("affine")
rigid_map = sitk.GetDefaultParameterMap("rigid")
bspline_map = sitk.GetDefaultParameterMap("bspline")

# Set the default parameter maps
paramVector = sitk.VectorOfParameterMap()
paramVector.append(affine_map)
paramVector.append(rigid_map)
paramVector.append(bspline_map)

# Preparation for splitting over train/validation/test
n_files = len(rho_files)
train_ratio = 0.7
validation_ratio = 0.1
n_train = int(train_ratio * n_files)
n_validation = int(validation_ratio * n_files)

train_files = rho_files[:n_train]
validation_files = rho_files[n_train:n_validation]
test_files = rho_files[n_validation:]

i_count = 0
N_files = None
dice_metrdice_metric_list = []
dice_treshold = 0.1

for i_rho_file in sorted(rho_files[0:N_files]):
    print(f"File.. {i_count + 1}/{n_files} ...")
    i_count += 1

    # Define in which folder the target data will land
    if i_rho_file in train_files:
        appendix_train = 'train'
    elif i_rho_file in validation_files:
        appendix_train = 'validation'
    else:
        appendix_train = 'test'

    print(f'\t Type: {appendix_train}')

    # Loading rho file
    rho_file_path = os.path.join(dir_rho, i_rho_file)
    temp_rho = np.load(rho_file_path)
    temp_rho = np.rot90(temp_rho)

    # Rho mask creation
    n_kernel = int(min(temp_rho.shape[-2:]) * 0.05)
    rho_mask_orig = harray.get_smoothed_mask(temp_rho, treshold_smooth=0.3,
                                             n_mask=n_kernel,
                                             conv_boundary='fill',
                                             conv_mode='valid')

    rho_mask_orig_shape = rho_mask_orig.shape
    shape_diff = (np.array(temp_rho.shape) - np.array(rho_mask_orig.shape)) // 2
    # Correct for the difference in shape....
    temp_rho = temp_rho[shape_diff[0]:rho_mask_orig_shape[0] + shape_diff[0],
                        shape_diff[1]:rho_mask_orig_shape[1] + shape_diff[1]]

    # Set the default parameter maps
    paramVector = sitk.VectorOfParameterMap()

    """
        A rigid transformation (also called an isometry) is a transformation of the plane that preserves length
    """
    rigid_map = sitk.GetDefaultParameterMap("rigid")
    paramVector.append(rigid_map)

    """
        Affine transformation is a linear mapping method that preserves points, straight lines, and planes. 
    """
    affine_map = sitk.GetDefaultParameterMap("affine")
    paramVector.append(affine_map)

    """
        A non-rigid transformation can change the size or shape, or both size and shape, of the preimage. 
        Two transformations, dilation and shear, are non-rigid. 
    """
    bspline_map = sitk.GetDefaultParameterMap("bspline")
    paramVector.append(bspline_map)

    """
    Store the mask of the rho density and the image itself
    """

    # rho  mask
    rho_nr = re.findall('([0-9]+)', i_rho_file)[0]
    concat_filename = '_'.join([rho_nr.zfill(2)]) + '.npy'
    dest_file_mask = os.path.join(dest_dir, appendix_train, 'mask', concat_filename)
    np.save(dest_file_mask, rho_mask_orig)

    # Rho file
    rho_nr = re.findall('([0-9]+)', i_rho_file)[0]
    concat_filename = '_'.join([rho_nr.zfill(2)]) + '.npy'
    dest_file_target_clean = os.path.join(dest_dir, appendix_train, 'target_clean', concat_filename)
    np.save(dest_file_target_clean, temp_rho)

    for i_b1_minus_file in b1_minus_files[0:(N_files)]:
        file_name_string = ', '.join([i_b1_minus_file, i_rho_file])
        # Loading b1 minus file
        b1_minus_file_path = os.path.join(dir_b1_minus, i_b1_minus_file)
        temp_b1_minus = np.load(b1_minus_file_path)

        n_kernel = int(min(temp_b1_minus.shape[-2:]) * 0.2)
        b1_minus_mask_orig = harray.get_smoothed_mask(np.abs(temp_b1_minus).sum(axis=0), treshold_smooth=0.9,
                                                      n_mask=n_kernel,
                                                      conv_boundary='fill',
                                                      conv_mode='valid')

        if plot_intermediate_results:
            hplotf.plot_3d_list([b1_minus_mask_orig, rho_mask_orig], title=file_name_string)

        # Transform b1 minus
        b1_minus_transform, min_scale_minus = register_mask(x_orig=b1_minus_mask_orig,
                                                            y_orig=rho_mask_orig,
                                                            param_vector=paramVector, debug=False)
        res_b1_minus, _ = validate_mask_mapping(b1_minus_mask_orig, transform_mapping=b1_minus_transform,
                                                min_scale=min_scale_minus, y_orig=rho_mask_orig, plot=False)
        b1_minus2rho_dice = hmisc.dice_metric(res_b1_minus, rho_mask_orig)

        if plot_intermediate_results:
            print('Dice metric b1 minus to rho ', b1_minus2rho_dice)
            hplotf.plot_3d_list([res_b1_minus, rho_mask_orig])

        # Store dice-metric values for later use
        dice_pass_or_no = b1_minus2rho_dice < dice_treshold
        temp_dict = {file_name_string: b1_minus2rho_dice}
        dice_metric_list.append(temp_dict)

        json_dump = json.dumps(temp_dict)
        with open(os.path.join(base_dir, 'dicescore.json'), 'a') as f:
            f.writelines(json_dump + '\n')

        """
        Apply everything on a multi coil input..
        """
        if dice_pass_or_no:
            # We are not going to execute this yet...
            n_chan, n_y, n_x = temp_b1_minus.shape
            stacked_b1_minus = harray.to_stacked(temp_b1_minus, cpx_type='cartesian', stack_ax=0).T
            stacked_b1_minus = stacked_b1_minus.reshape((n_x, n_y, -1)).T

            stacked_b1_minus = apply_transform(stacked_b1_minus, min_scale=min_scale_minus,
                                              transform_mapping=b1_minus_transform,
                                              mask=rho_mask_orig)

            if plot_intermediate_results:
                hplotf.plot_3d_list(stacked_b1_minus[None])

            # Output stuff...
            n_y, n_x = stacked_b1_minus[0].shape
            input = np.moveaxis(stacked_b1_minus, 0, -1).reshape((n_y, n_x, n_chan, 2)).T.reshape(2 * n_chan, n_x, n_y).T
            b1_minus_registered = harray.to_complex_chan(input, img_shape=(n_y, n_x), n_chan=n_chan,
                                                        complex_type='cartesian')

            if plot_intermediate_results:
                hplotf.plot_3d_list(b1_minus_registered, augm='np.abs')

            # B1 minus files
            rho_nr = re.findall('([0-9]+)', i_rho_file)[0]
            b1min = re.findall('([0-9]{8})', i_b1_minus_file)[0]
            concat_filename = '_'.join([rho_nr.zfill(2), b1min]) + '.npy'
            dest_file_input = os.path.join(dest_dir, appendix_train, 'input', concat_filename)
            np.save(dest_file_input, b1_minus_registered[0])

    for i_b1_plus_file in b1_plus_files[0:N_files]:
        file_name_string = ', '.join([i_b1_plus_file, i_rho_file])
        # Loading b1 plus file...
        b1_plus_file_path = os.path.join(dir_b1_plus, i_b1_plus_file)
        temp_b1_plus = np.load(b1_plus_file_path)

        n_kernel = int(min(temp_b1_plus.shape[-2:]) * 0.1)
        b1_plus_mask_orig = harray.get_smoothed_mask(np.abs(temp_b1_plus).sum(axis=0), treshold_smooth=0.8,
                                                     n_mask=n_kernel,
                                                     conv_boundary='fill',
                                                     conv_mode='valid')

        if plot_intermediate_results:
            hplotf.plot_3d_list([b1_plus_mask_orig, rho_mask_orig], title=file_name_string)

        # Transform b1 plus
        b1_plus_transform, min_scale_plus = register_mask(x_orig=b1_plus_mask_orig,
                                                          y_orig=rho_mask_orig,
                                                          param_vector=paramVector, debug=False)
        res_b1_plus, _ = validate_mask_mapping(b1_plus_mask_orig, transform_mapping=b1_plus_transform,
                                               min_scale=min_scale_plus, y_orig=rho_mask_orig, plot=False)
        b1_plus2rho_dice = hmisc.dice_metric(res_b1_plus, rho_mask_orig)

        if plot_intermediate_results:
            print('Dice metric b1 minus to rho ', b1_plus2rho_dice)
            hplotf.plot_3d_list([res_b1_plus, rho_mask_orig])

        # Store dice-metric values for later use
        dice_pass_or_no = b1_plus2rho_dice < dice_treshold
        temp_dict = {file_name_string: b1_plus2rho_dice}
        dice_metric_list.append(temp_dict)

        json_dump = json.dumps(temp_dict)
        with open(os.path.join(base_dir, 'dicescore.json'), 'a') as f:
            f.writelines(json_dump + '\n')

        """
        Apply everything on a multi coil input..
        """
        if dice_pass_or_no:
            # We are not going to execute this yet...
            n_chan, n_y, n_x = temp_b1_plus.shape
            stacked_b1_plus = harray.to_stacked(temp_b1_plus, cpx_type='cartesian', stack_ax=0).T
            stacked_b1_plus = stacked_b1_plus.reshape((n_x, n_y, -1)).T

            stacked_b1_plus = apply_transform(stacked_b1_plus, min_scale=min_scale_plus,
                                              transform_mapping=b1_plus_transform,
                                              mask=rho_mask_orig)

            if plot_intermediate_results:
                hplotf.plot_3d_list(stacked_b1_plus[None])

            # Output stuff...
            n_y, n_x = stacked_b1_plus[0].shape
            input = np.moveaxis(stacked_b1_plus, 0, -1).reshape((n_y, n_x, n_chan, 2)).T.reshape(2 * n_chan, n_x, n_y).T
            b1_plus_registered = harray.to_complex_chan(input, img_shape=(n_y, n_x), n_chan=n_chan,
                                                        complex_type='cartesian')

            if plot_intermediate_results:
                hplotf.plot_3d_list(b1_plus_registered, augm='np.angle')

            """
            Store results
            """
            rho_nr = re.findall('([0-9]+)', i_rho_file)[0]
            b1plus = re.findall('([0-9]+)', i_b1_plus_file)[0]
            concat_filename = '_'.join([rho_nr.zfill(2), b1plus.zfill(2)]) + '.npy'
            # B1 plus files
            dest_file_target = os.path.join(dest_dir, appendix_train, 'target', concat_filename)
            np.save(dest_file_target, b1_plus_registered[0])
