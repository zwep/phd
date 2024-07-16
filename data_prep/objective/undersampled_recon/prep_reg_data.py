# encoding: utf-8


"""
Here we only do the registration.. not the undersampling. This gives us 8x8 combined images.

This was the start of a simple undersampling project...

"""

import numpy as np
import os
import SimpleITK as sitk
import nrrd
import re
import sys
from pynufft import NUFFT_cpu


def _register_and_transform(input_array, transform_map, mask):
    # Used to automate the registration for b1plus and b1minus..
    # Apply transformation
    x_registered = []
    # Probably loops over the amount of coils...
    for x in input_array:
        x_image = sitk.GetImageFromArray(x)
        result_mapping = sitk.Transformix(x_image, transform_map)
        x_array = sitk.GetArrayFromImage(result_mapping)
        x_array = x_array * mask
        # Could use this instead.... (for the Angle correction)
        mask_int = ((1 - mask) == 1).astype(int)
        input_array_close = np.isclose(x_array, 0).astype(int)
        input_array_outside = (input_array_close * mask_int).astype(bool)
        x_array[input_array_outside] = 0
        # This was the old way...
        # x_array[np.isclose(x_array, 0)] = 0
        x_registered.append(x_array)

    x_registered = np.array(x_registered)
    n_y, n_x = x_registered[0].shape
    input = np.moveaxis(x_registered, 0, -1).reshape(n_y, n_x, 8, 2).T.reshape(16, n_x, n_y).T
    n_chan = 8
    result = harray.to_complex_chan(input, img_shape=(n_y, n_x), n_chan=n_chan, complex_type='cartesian')
    return result

import getpass

# Deciding which OS is being used
if getpass.getuser() == 'bugger':
    local_system = True
    manual_mode = True
    # dest_dir = '/home/bugger/Documents/data/semireal/prostate_simulation'
    dest_dir = '/media/bugger/MyBook/data/simreal/prostate_simulation'
    dir_data_flavio = '/home/bugger/Documents/data/simulation/flavio_npy'
    data_dir = '/home/bugger/Documents/data/prostatemriimagedatabase'

    project_path = "/home/bugger/PycharmProjects/pytorch_in_mri"
else:
    import matplotlib as mpl
    mpl.use('Agg')  # Hopefully this makes sure that we can plot/save stuff
    local_system = False
    manual_mode = False
    dest_dir = '/data/seb/semireal/prostate_simulation_rxtx'
    dir_data_flavio = '/data/seb/flavio_npy'
    data_dir = '/data/seb/prostatemriimagedatabase'
    project_path = "/home/seb/code/pytorch_in_mri"


model_path = "/home/seb/code/pytorch_in_mri"
sys.path.append(project_path)

import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
import data_generator.Rx2Tx as gen_rx2tx
import helper.nufft_recon as hnufft

hmisc.create_datagen_dir(dest_dir, data_list=['input', 'mask', 'transform_map', 'target'])
# These are excluded because they involved arms, weird anatomical shapres, or ill defined boundaries
exclude_file = ['000019.00002.005.nrrd', '000016.00002.006.nrrd', '000012.00002.005.nrrd']
file_list = sorted([x for x in os.listdir(data_dir) if x.endswith('.nrrd') and (x not in exclude_file)])

n_files = len(file_list)
train_ratio = 0.7
n_train = int(train_ratio * n_files)
train_files = file_list[:n_train]

plot_intermediate_results = False

n_files = len(file_list)
# This is saved locally. Now we are going to store it in MyBook
for i_count, i_file in enumerate(file_list):
    print(f"File.. {i_count + 1}/{n_files} ...")

    file_path = os.path.join(data_dir, i_file)
    temp_data, temp_header = nrrd.read(file_path)
    temp_data = np.moveaxis(temp_data, -1, 0)

    file_name_prostate, ext = os.path.splitext(i_file)
    file_name_prostate = re.sub('\.', '_', file_name_prostate)

    if i_file in train_files:
        train_id = True
        appendix_train = 'train'
    else:
        train_id = False
        appendix_train = 'test'

    print(f'\t Train: {train_id}')

    if plot_intermediate_results:
        # Inspect data itself
        n_img = temp_data.shape[0]
        n_step = n_img//20
        hplotf.plot_3d_list(temp_data[np.newaxis, ::n_step], ax_off=True, title=i_file)

        # Inspect mask creation
        temp_mask = []
        temp_image = []
        for i_prostate_array in temp_data:
            input_prostate = np.flipud(np.rot90(i_prostate_array))
            prostate_mask = harray.get_smoothed_mask(input_prostate, treshold_smooth=0.8)
            temp_mask.append(prostate_mask)
            temp_image.append(input_prostate)

        temp_mask = np.array(temp_mask)
        temp_image = np.array(temp_image)
        n_img = temp_mask.shape[0]
        n_step = n_img//20
        hplotf.plot_3d_list(temp_image[np.newaxis, ::n_step], ax_off=True, title=i_file)
        hplotf.plot_3d_list(temp_mask[np.newaxis, ::n_step], ax_off=True, title=i_file)

    # Continue...
    n_slices = len(temp_data)
    for i_slice, i_prostate_array in enumerate(temp_data):
        print(f"slice  {i_slice + 1}/{n_slices} ...")
        file_name_prostate_slice = file_name_prostate + f'__{i_slice}'
        # Fix orientation of the input image...
        input_prostate = np.flipud(np.rot90(i_prostate_array))
        prostate_mask = harray.get_smoothed_mask(input_prostate, treshold_smooth=0.8)
        dest_file_mask = os.path.join(dest_dir, appendix_train, 'mask', file_name_prostate_slice)
        np.save(dest_file_mask, prostate_mask)

        # Load the data generator
        dg_gen_rx2tx_flavio = gen_rx2tx.DataSetSurvey2B1_flavio(input_shape=(2, 512, 256),
                                                                ddata=dir_data_flavio,
                                                                masked=True,
                                                                train=train_id)

        n_files_flavio = len(dg_gen_rx2tx_flavio)
        for i_flavio_array in range(n_files_flavio):
            print(f'\t Loading flavio array number {i_flavio_array}')
            b1_minus, b1_plus, flavio_mask = dg_gen_rx2tx_flavio.__getitem__(i_flavio_array)

            if plot_intermediate_results:
                # This is for the presentation
                b1_minus_np = b1_minus.numpy()
                coil_1 = b1_minus_np[4] + 1j * b1_minus_np[5]
                hplotf.plot_3d_list(np.angle(coil_1)[np.newaxis], title='7T B1-minus distribution, coil 2',
                                    ax_off=True, cbar=True)

                b1_plus_np = b1_plus.numpy()
                coil_1 = b1_plus_np[4] + 1j * b1_plus_np[5]
                hplotf.plot_3d_list(np.angle(coil_1)[np.newaxis], title='7T B1-plus distribution, coil 2',
                                    ax_off=True, cbar=True)

                hplotf.plot_3d_list(input_prostate[np.newaxis], title='1.5T Prostate image',
                                    ax_off=True, cbar=True)

                hplotf.plot_3d_list([flavio_mask.numpy().astype(int)], cbar=True, ax_off=True)
                hplotf.plot_3d_list([prostate_mask], cbar=True, ax_off=True)

            file_name_flavio = dg_gen_rx2tx_flavio.file_list[i_flavio_array]
            file_name_flavio, _ = os.path.splitext(file_name_flavio)

            dest_file_name = file_name_flavio + '_to_' + file_name_prostate_slice

            # It has a 1 dimension on axis position 0..
            flavio_mask_image = sitk.GetImageFromArray(flavio_mask[0])
            prostate_mask_image = sitk.GetImageFromArray(prostate_mask)

            elastixImageFilter = sitk.ElastixImageFilter()
            elastixImageFilter.LogToConsoleOff()  # Chilll
            elastixImageFilter.SetFixedImage(prostate_mask_image)
            elastixImageFilter.SetMovingImage(flavio_mask_image)

            parameterMapVector = sitk.VectorOfParameterMap()
            parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
            parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
            elastixImageFilter.SetParameterMap(parameterMapVector)

            result_image = elastixImageFilter.Execute()
            result_array = sitk.GetArrayFromImage(result_image)

            transform_map = elastixImageFilter.GetTransformParameterMap()

            # TODO .. store this map..? later
            # Apply transformation to real/imag of all 8 coils
            if plot_intermediate_results:
                validate_image = sitk.Transformix(flavio_mask_image, transform_map)
                validate_array = sitk.GetArrayFromImage(validate_image)
                hplotf.plot_3d_list([result_array, validate_array, result_array - validate_array])

            b1_minus_registered = _register_and_transform(b1_minus, transform_map=transform_map, mask=prostate_mask)
            b1_plus_registered = _register_and_transform(b1_plus, transform_map=transform_map, mask=prostate_mask)

            if plot_intermediate_results:
                hplotf.plot_3d_list(b1_minus_registered, augm='np.angle')
                hplotf.plot_3d_list(b1_plus_registered, augm='np.angle')

            dest_file_input = os.path.join(dest_dir, appendix_train, 'input', dest_file_name)
            dest_file_target = os.path.join(dest_dir, appendix_train, 'target', dest_file_name)

            # This is still saving as numpy array...
            # If I ever want to use this code again, I should translate this to an .h5 solution.
            # Which just means.. concat all the data into one big high dimensional array
            # And then store it.
            np.save(dest_file_input, b1_minus_registered)
            np.save(dest_file_target, b1_plus_registered)


if __name__ == "__main__":
    # Testing whether we did it right in the previous case..
    import numpy as np
    import nrrd
    import matplotlib.pyplot as plt
    import helper.array_transf as harray

    aa = '/home/bugger/Documents/data/simulation/prostate/train/input/Duke.npy'
    aa_mask = '/home/bugger/Documents/data/simulation/prostate/train/mask/Duke.npy'
    bb = '/home/bugger/Documents/data/prostateimagedatabase/000003.00002.005.nrrd'

    A = np.load(aa)
    A_mask = np.load(aa_mask)
    B = nrrd.read(bb)

    plt.imshow(np.abs(A)[0])
    plt.imshow(B[0][:, :, 0])

    input_prostate = np.flipud(np.rot90(B[0][:, :, 0]))
    prostate_mask = harray.get_smoothed_mask(input_prostate, treshold_smooth=0.8)

    plt.imshow(prostate_mask)
    plt.imshow(A_mask)

    import SimpleITK as sitk
    import helper.plot_fun as hplotf
    flavio_mask_image = sitk.GetImageFromArray(A_mask)
    prostate_mask_image = sitk.GetImageFromArray(prostate_mask)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()  # Chilll
    # elastixImageFilter.SetFixedImage(prostate_mask_image)
    # elastixImageFilter.SetMovingImage(flavio_mask_image)
    elastixImageFilter.SetFixedImage(flavio_mask_image)
    elastixImageFilter.SetMovingImage(prostate_mask_image)

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    elastixImageFilter.SetParameterMap(parameterMapVector)

    result_image = elastixImageFilter.Execute()
    registered_plus_mask_image = elastixImageFilter.GetResultImage()

    result_array = sitk.GetArrayFromImage(result_image)

    hplotf.plot_3d_list([result_array, A_mask, prostate_mask])

    tfMap = elastixImageFilter.GetTransformParameterMap()
    result_mapping = sitk.Transformix(prostate_mask_image, tfMap)
    result_array3 = sitk.GetArrayFromImage(result_mapping)

    hplotf.plot_3d_list([result_array3, A_mask, prostate_mask])