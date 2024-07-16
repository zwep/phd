
"""
Here we actually register the images/masks together.

We have the following masks
    - b1 minus
    - b1 plus
    - rho

We need to transform everything onto the real magnitude images to preserve detail


I think the best thing is to create a mask on both Rho images..
Use that transformation map on the b1plus and b1minus maps

Split the result-files into train/test/validation based on
weighted ID separation

Loop over all the result-files.
    Get array sim
        rho
            Make square
    Get array real
        rho

    Create mask sim-rho
        Then resize to size of real-rho
    Create mask real-rho

    Apply registration to map sim-rho to real-rho

    Get array sim
        b1 plus
        b1 minus
    Apply transformation

"""

import SimpleITK as sitk
import getpass
import json
import numpy as np
import skimage.transform as sktrans
import sys
import os
import re
import collections

if getpass.getuser() == 'bugger':
    local_system = True
    project_path = '/'

else:
    import matplotlib as mpl
    mpl.use('Agg')  # Hopefully this makes sure that we can plot/save stuff
    local_system = False
    project_path = '/home/seb/code/pytorch_in_mri'

sys.path.append(project_path)

import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc


# This is about the masks....
def register_mask(x_orig, y_orig, param_vector, debug=False, mask=False):
    """We map x_orig to y_orig.."""
    elastix_obj = sitk.ElastixImageFilter()
    if not debug:
        elastix_obj.LogToConsoleOff()

    # Convert to Image object
    moving_image = sitk.GetImageFromArray(x_orig.astype(int))
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
        hplotf.plot_3d_list([transformed_array, x_orig, y_orig],
                            subtitle=[['transformed'], ['moving'], ['fixed']])

    transform_mapping = elastix_obj.GetTransformParameterMap()
    return transform_mapping


def validate_mask_mapping(x_orig, transform_mapping, y_orig=None, plot=False):
    # With this we normally check again if the transformation went as we want
    x_image = sitk.GetImageFromArray(x_orig)
    validate_image = sitk.Transformix(x_image, transform_mapping)
    validate_array = sitk.GetArrayFromImage(validate_image)

    dice_score = -1
    if plot:
        if y_orig is not None:
            dice_score = hmisc.dice_metric(y_orig, validate_array)
            fig_handle = hplotf.plot_3d_list([validate_array, x_orig, y_orig],
                                subtitle=[['transformed'], ['moving'], ['target']], title=dice_score)
        else:
            fig_handle = hplotf.plot_3d_list([validate_array, x_orig])
    else:
        fig_handle = -1

    return validate_array, dice_score, fig_handle


def apply_transform(x_orig, transform_mapping, mask=None):
    # Apply the transform-map to x_orig
    # x_orig can be a ndarray, we loop over the first dimension
    # We correct with the given mask to make sure it fits all well
    if x_orig.ndim == 2:
        x_orig = x_orig[np.newaxis]
    else:
        print('Input dimension is ', x_orig.ndim)

    if 'complex' in str(x_orig.dtype):
        # Do I really want this behaviour..?
        print('Warning! Input is complex. Continuing with real values')

    temp = []
    for x in x_orig:
        x_image = sitk.GetImageFromArray(x)
        x_transformed = sitk.Transformix(x_image, transform_mapping)
        x_array = sitk.GetArrayFromImage(x_transformed)

        if mask is not None:
            # Correct with mask
            # Make sure that we have no troubles with really tiny numbers
            x_array = x_array * mask
            mask_int = ((1 - mask) == 1).astype(int)
            input_array_close = np.isclose(x_array, 0).astype(int)
            input_array_outside = (input_array_close * mask_int).astype(bool)
            x_array[input_array_outside] = 0

        temp.append(x_array)

    temp_stacked = np.stack(temp, axis=0)
    return temp_stacked


plot_intermediate_results = False
filter_results = False

# Change paths according to the station we are on...
if local_system:
    simulation_base_dir = '/home/bugger/Documents/data/simulation'
    real_base_dir = '/home/bugger/Documents/data/1.5T'
    dest_base_dir = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx'
else:
    # I think I have to create these still...
    simulation_base_dir = '/data/seb/simulation'
    real_base_dir = '/data/seb/1.5T'
    dest_base_dir = '/data/seb/semireal/cardiac_simulation_rxtx'

scan_type = 'p2ch' # Either p2ch or 4ch
sim_plus_dir = f'{simulation_base_dir}/cardiac/b1/{scan_type}/b1_plus'
sim_minus_dir = f'{simulation_base_dir}/cardiac/b1/{scan_type}/b1_minus'
sim_rho_dir = f'{simulation_base_dir}/cardiac/b1/{scan_type}/rho'
sim_results_dir = f'{simulation_base_dir}/cardiac/b1/{scan_type}/results'
real_dir = f'{real_base_dir}/{scan_type}'
dest_dir = f'{dest_base_dir}/{scan_type}_registered'

result_files = os.listdir(sim_results_dir)

hmisc.create_datagen_dir(dest_dir, type_list=['train', 'test', 'validation'],
                         data_list=['input', 'mask', 'target', 'target_clean'])

"""
Here we try to filter the results and split it properly..

Training set:
    V1 - V10 combined with scan Ids [0-6]...
Test set:
    V11 - V14 combined with scan Ids [7-9][0-8]... 
Validation set:
    V14 combined with scan IDs 99...
"""

re_cardiac_train = re.compile('Array_V[1-9](0|)_')
re_scan_train = re.compile('Data_[0-6]')

re_cardiac_validation = re.compile('Array_V14')
re_scan_validation = re.compile('Data_99')

re_cardiac_test = re.compile('Array_V1[0-3]')
re_scan_test = re.compile('Data_[7-9][0-8]')

train_files = [x for x in result_files if re_cardiac_train.findall(x) and re_scan_train.findall(x)]
validation_files = [x for x in result_files if re_cardiac_validation.findall(x) and re_scan_validation.findall(x)]
test_files = [x for x in result_files if re_cardiac_test.findall(x) and re_scan_test.findall(x)]

all_files = train_files + validation_files + test_files
all_files = [os.path.join(sim_results_dir, x) for x in all_files]

"""
Filter on files we have already processed
"""
registered_file_list = []
for d_dir, _, f_files in os.walk(dest_dir):
    if f_files:
        registered_file_list.extend(f_files)

# Double check if we really want to filter it...
if filter_results:
    print('We have already completed the following files: ')
    [print(x) for x in registered_file_list]
    print('We are filtering', len(registered_file_list))
    registered_file_list = [os.path.splitext(x)[0] for x in registered_file_list]

    n_unfiltered = len(all_files)
    print('We are starting with ', n_unfiltered)
    sel_all_files = [x for x in all_files if os.path.splitext(os.path.basename(x))[0] not in registered_file_list]

    print('We are left with ', len(sel_all_files))
else:
    print('We are left with ', len(all_files))
    sel_all_files = all_files

"""
Start the registration loop
"""
file_counter = 0
for i_results_file in sel_all_files:
    file_counter += 1
    file_name = os.path.basename(i_results_file)
    print('Starting with file ', file_name, file_counter, '/', len(sel_all_files))

    if file_name in train_files:
        type_prefix = 'train'
    elif file_name in test_files:
        type_prefix = 'test'
    elif file_name in validation_files:
        type_prefix = 'validation'
    else:
        type_prefix = None

    with open(i_results_file, 'r') as f:
        temp = f.read()
        config_results = json.loads(temp)

    # Collect the names for the density files, and mask file of real-rho
    scan_id = re.findall('[0-9]{7}', i_results_file)[0]

    sim_file_name = config_results['sim']['sel_file_name']
    real_file_name = config_results['real']['sel_file_name']
    # We split this to create the name for the mask file more easily
    real_file_name, file_ext = os.path.splitext(real_file_name)

    # Path for the original density image..
    dest_file_name = os.path.splitext(file_name)[0]
    dest_file_input = os.path.join(dest_dir, type_prefix, 'input', dest_file_name)
    dest_file_target = os.path.join(dest_dir, type_prefix, 'target', dest_file_name)
    dest_file_mask = os.path.join(dest_dir, type_prefix, 'mask', dest_file_name)
    dest_file_target_clean = os.path.join(dest_dir, type_prefix, 'target_clean', dest_file_name)

    """
    Load the density images
    """
    real_rho = np.load(os.path.join(real_dir, real_file_name + file_ext))
    sim_rho = np.load(os.path.join(sim_rho_dir, sim_file_name))
    sim_rho = np.squeeze(sim_rho)
    print(real_rho.shape, sim_rho.shape)

    # Adapt real_rho to get a nice 2d array
    if config_results['real']['flipped_ud']:
        real_rho = real_rho[:, ::-1]

    if config_results['real']['flipped_lr']:
        real_rho = real_rho[:, :, ::-1]

    if real_rho.ndim == 3:
        sel_time = real_rho.shape[0] // 2
        real_rho = real_rho[sel_time]

    co_y = config_results['sim']['coordinates0']
    co_x = config_results['sim']['coordinates1']

    # Adapt sim_rho to sub-select the matrix, using the results
    print('Coordinates of subsampled')
    print('\t', co_y, co_x)
    sel_sim_rho = sim_rho[co_y[0]:co_y[1], co_x[0]:co_x[1]]
    # Make sure that we have a square matrix
    n_x, n_y = sel_sim_rho.shape[-2:]
    n = min(n_x, n_y)
    sel_sim_rho = sel_sim_rho[:n, :n]

    # Plot the current loaded density images
    if plot_intermediate_results:
        hplotf.plot_3d_list([sel_sim_rho, real_rho], augm='np.abs')

    """
    Load/create the mask images for the density images
    """
    # Load the mask image for the real array
    try:
        mask_real_rho = np.load(os.path.join(real_dir, 'mask', real_file_name + '_mask' + file_ext))
    except FileNotFoundError:
        pass
        # continue

    # Adapt real_rho to get a nice 2d array
    if config_results['real']['flipped_ud']:
        mask_real_rho = mask_real_rho[::-1]

    if config_results['real']['flipped_lr']:
        mask_real_rho = mask_real_rho[:, ::-1]

    # Create the mask for the sub selected image
    treshold_smooth = 0.6
    kernel_factor = 0.1
    debug_intermediate = False
    n_kernel = int(n * kernel_factor)
    mask_sim_rho = harray.get_smoothed_mask(sel_sim_rho, treshold_smooth=treshold_smooth,
                                            n_mask=n_kernel, debug=debug_intermediate,
                                            conv_boundary='fill',
                                            conv_mode='valid')

    sim_shape = sim_rho.shape
    sim_xy_shape = sim_rho.shape[-2:]

    # Resize the mask images to comply with the original shape
    # The get_smoothed_mask can reduce the images size
    resized_sim_mask = sktrans.resize(mask_sim_rho, real_rho.shape, order=3, preserve_range=True)
    resized_real_mask = sktrans.resize(mask_real_rho.astype(int), real_rho.shape, order=3, preserve_range=True)

    # Plot the masks we just loaded/created
    if plot_intermediate_results:
        hplotf.plot_3d_list([resized_sim_mask, resized_real_mask])

    """
    Create the parameter vector map for the registration
    """

    # Set the default parameter maps
    paramVector = sitk.VectorOfParameterMap()

    """
        A rigid transformation (also called an isometry) is a transformation of the plane that preserves length
    """
    rigid_map = sitk.GetDefaultParameterMap("rigid")
    if scan_type == 'p4ch':
        paramVector.append(rigid_map)
    elif scan_type == 'p2ch':
        pass


    """
        Affine transformation is a linear mapping method that preserves points, straight lines, and planes. 
    """
    affine_map = sitk.GetDefaultParameterMap("affine")
    if scan_type == 'p4ch':
        paramVector.append(affine_map)
    elif scan_type == 'p2ch':
        pass


    """
        A non-rigid transformation can change the size or shape, or both size and shape, of the preimage. 
        Two transformations, dilation and shear, are non-rigid. 
    """
    bspline_map = sitk.GetDefaultParameterMap("bspline")
    paramVector.append(bspline_map)


    """
    Perform registration on the rho-masks from sim to real
    """
    # derp = sktrans.rescale(resized_sim_mask, scale=1.6, preserve_range=True)

    result_mapping = register_mask(resized_sim_mask, resized_real_mask, paramVector, debug=debug_intermediate)

    """
    Validate the mapping and calculate a dice-score for the fit...
    """
    validate_array, dice_score, fig_handle = validate_mask_mapping(x_orig=resized_sim_mask, y_orig=resized_real_mask,
                                                                   transform_mapping=result_mapping, plot=True)
    dice_pass_or_no = False
    if dice_score < 0.1:
        dice_pass_or_no = True

    print('Status of dice score ', dice_pass_or_no)

    """
    Apply everything on a multi coil input..
    """
    # This one comes from the yay or nay from the sim rho to real rho mask transformation

    if dice_pass_or_no:
        """
        Transform the b1-minus files 
        """
        sim_b1_minus = np.load(os.path.join(sim_minus_dir, sim_file_name))

        # We are not going to execute this yet...
        n_chan, n_y, n_x = sim_b1_minus.shape
        stacked_b1_minus = harray.to_stacked(sim_b1_minus, cpx_type='cartesian', stack_ax=0).T
        stacked_b1_minus = stacked_b1_minus.reshape((n_x, n_y, -1)).T

        # Sub-select the image, just as we did with sim-rho
        stacked_b1_minus = stacked_b1_minus[:, co_y[0]:co_y[1], co_x[0]:co_x[1]]
        n_y, n_x = stacked_b1_minus.shape[-2:]
        n = min(n_y, n_x)
        stacked_b1_minus = stacked_b1_minus[:, :n, :n]
        temp_chan = stacked_b1_minus.shape[0]

        # Re-scale it to its original shape
        resized_stacked_b1_minus = sktrans.resize(stacked_b1_minus, output_shape=(temp_chan, ) + real_rho.shape,
                                                  order=3, preserve_range=True)

        stacked_b1_minus = apply_transform(x_orig=resized_stacked_b1_minus,
                                           transform_mapping=result_mapping,
                                           mask=resized_real_mask)

        # Update the new n_y and n_x of the image..
        n_y, n_x = resized_real_mask.shape

        stacked_b1_minus = np.moveaxis(stacked_b1_minus, 0, -1).reshape((n_y, n_x, n_chan, 2)).T.reshape(2 * n_chan, n_x, n_y).T
        b1_minus_registered = harray.to_complex_chan(stacked_b1_minus, img_shape=(n_y, n_x), n_chan=n_chan, complex_type='cartesian')

        if plot_intermediate_results:
            hplotf.plot_3d_list(b1_minus_registered, augm='np.abs')

        """
        Transform the b1-plus files
        """
        sim_b1_plus = np.load(os.path.join(sim_plus_dir, sim_file_name))

        if plot_intermediate_results:
            hplotf.plot_3d_list(sim_b1_plus[None], augm='np.abs', title='Orignal of b1 plus')

        # We are not going to execute this yet...
        n_chan, n_y, n_x = sim_b1_plus.shape
        stacked_b1_plus = harray.to_stacked(sim_b1_plus, cpx_type='cartesian', stack_ax=0).T
        stacked_b1_plus = stacked_b1_plus.reshape((n_x, n_y, -1)).T

        # Sub-select the image, just as we did with sim-rho
        stacked_b1_plus = stacked_b1_plus[:, co_y[0]:co_y[1], co_x[0]:co_x[1]]
        n_y, n_x = stacked_b1_plus.shape[-2:]
        n = min(n_y, n_x)
        stacked_b1_plus = stacked_b1_plus[:, :n, :n]
        temp_chan = stacked_b1_plus.shape[0]

        # Re-scale it to its original shape
        resized_stacked_b1_plus = sktrans.resize(stacked_b1_plus, output_shape=(temp_chan, ) + real_rho.shape,
                                                 order=3, preserve_range=True)

        stacked_b1_plus = apply_transform(x_orig=resized_stacked_b1_plus,
                                          transform_mapping=result_mapping,
                                          mask=resized_real_mask)

        n_y, n_x = resized_real_mask.shape
        stacked_b1_plus = np.moveaxis(stacked_b1_plus, 0, -1).reshape((n_y, n_x, n_chan, 2)).T.reshape(2 * n_chan, n_x, n_y).T
        b1_plus_registered = harray.to_complex_chan(stacked_b1_plus, img_shape=(n_y, n_x), n_chan=n_chan,
                                                    complex_type='cartesian')

        if plot_intermediate_results:
            hplotf.plot_3d_list(b1_plus_registered, augm='np.abs', vmin=(0, np.abs(b1_plus_registered).max()))

        result_image = b1_plus_registered * b1_minus_registered * real_rho
        result_image = result_image / np.max(result_image)

        if plot_intermediate_results:
            hplotf.plot_3d_list(result_image, augm='np.abs')

        """
        Store results
        """
        np.save(dest_file_input, b1_minus_registered)
        np.save(dest_file_target, b1_plus_registered)
        np.save(dest_file_target_clean, real_rho)
        np.save(dest_file_mask, resized_real_mask)
