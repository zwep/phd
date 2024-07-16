"""
I dont trust the way the data is created...
Here we create some code that can be run remotely to store some input examples...
"""
from skimage.util import img_as_ubyte
import data_generator.InhomogRemoval as data_gen
import helper.plot_class as hplotc
import numpy as np
import helper.array_transf as harray
import helper.misc as hmisc
import nibabel
import os

import numpy as np
import matplotlib.pyplot as plt

dir_data = '/local_scratch/sharreve/mri_data/registrated_h5'
dir_data = '/media/bugger/MyBook/data/registrated_h5'
ddest = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti'
ddest_input = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/input'
ddest_input_abs_sum = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/input_abs_sum'
ddest_target = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/target'
ddest_target_biasf = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/target_biasf'
ddest_mask = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/mask'

i_SNR = 5
for i_SNR in range(5, 30, 5):
    gen = data_gen.DataGeneratorInhomogRemovalH5(ddata=dir_data,
                                                 dataset_type='test', complex_type='cartesian',
                                                 file_ext='h5',
                                                 masked=True,
                                                 debug=False,
                                                 relative_phase=True,
                                                 transform_type='complex',
                                                 transform_type_target='real',
                                                 target_type='both',
                                                 shuffle=False,
                                                 cycle_all=False,
                                                 SNR_mode=i_SNR)
        #
    SNR_values = []
    container = gen.__getitem__(0)
    # hmisc.load_array(ddata, sel_slice=69)
    input_image = container['input'].numpy()
    rho_image = container['target'].numpy()[0]
    biasfield_image = container['target'].numpy()[1]
    mask_array = container['mask'].numpy()[0]
    abs_magn_image = (np.abs(input_image[::2] + 1j * input_image[1::2])).sum(axis=0)
    hplotc.ListPlot(abs_magn_image, title=i_SNR)

    squared_magn_image = (np.abs(input_image[::2] + 1j * input_image[1::2]) ** 2).sum(axis=0)
    squared_noise_image = (np.abs(input_image[::2] + 1j * input_image[1::2]) ** 2).sum(axis=0)[mask_array != 1]
    # To validate the std calculation of a sum of squares...
    2 * 0.5 / 100 * np.sqrt(8)
    np.sqrt(np.var(squared_noise_image))
    # Here we calculate sum of absolutes...

    abs_noise_image = (np.abs(input_image[::2] + 1j * input_image[1::2])).sum(axis=0)[mask_array != 1]
    #
    zi = np.abs(input_image[::2] + 1j * input_image[1::2])[0]
    np.var(zi[mask_array != 1])

    np.var(abs_noise_image)
    8 * (0.5 / 100) * (1 - 2 / np.pi) * np.sqrt(2)
    #
    0.5 /100 - np.var(abs_noise_image) / (np.sqrt(2) * 8)
    n_coils = 1
    sigma = 1
    imag_noise = np.random.multivariate_normal(np.zeros(n_coils),
                                               np.eye(n_coils) * sigma,
                                               size=(500, 500))
    np.var(np.abs(imag_noise))
    sigma ** 2 - (sigma * np.sqrt(2 / np.pi)) ** 2

    np.mean(abs_noise_image) ** 2


    print(1 / np.std(squared_noise_image))
    print(1 / np.std(abs_noise_image))
    fig_obj = hplotc.ListPlot((np.abs(input_image[::2] + 1j * input_image[1::2]) ** 2).sum(axis=0), title=i_SNR)
    fig_obj.figure.savefig(f'/local_scratch/sharreve/test_SNR_{i_SNR}.png')



container = gen.__getitem__(0)
input_image = container['input'].numpy()
mask_image = container['mask'].numpy()[0]
mask_image_shrink = harray.shrink_image(mask_image, 50)
input_cpx = input_image[::2] + 1j * input_image[1::2]
input_cpx = harray.scale_minmax(input_cpx, is_complex=True)

abs_sum = np.abs(input_cpx).sum(axis=0)
abs_sum_squared = np.sqrt((input_cpx * input_cpx.conjugate()).sum(axis=0))
fig_obj = hplotc.ListPlot([abs_sum, abs_sum_squared, mask_image_shrink], augm='np.abs', cbar=True, cbar_round_n=10)
fig_obj.figure.savefig('/local_scratch/sharreve/compared.png')

n_coils = input_cpx.shape[0]
imag_noise = real_noise = np.random.multivariate_normal(np.zeros(n_coils),
                                                        np.eye(n_coils),
                                                        size=input_cpx.shape[-2:])
noise_cpx = (real_noise + 1j * imag_noise).T
for i_snr in range(1, 20):
    measured_array = np.ones(input_cpx.shape) * i_snr / np.sqrt(n_coils) + noise_cpx
    measured_abs_sum_squared = np.sqrt((measured_array * measured_array.conjugate()).sum(axis=0))
    noise_abs_sum_squared = np.sqrt((noise_cpx * noise_cpx.conjugate()).sum(axis=0))
    #
    std_noise_estimate = np.mean(noise_abs_sum_squared ** 2) / np.sqrt(2 * n_coils)
    mean_signal_estimate = np.mean(measured_abs_sum_squared[mask_image_shrink == 1])
    print(mean_signal_estimate)
    fig_obj = hplotc.ListPlot([measured_abs_sum_squared], augm='np.abs', cbar=True, cbar_round_n=10)
    fig_obj.figure.savefig(f'/local_scratch/sharreve/meausred_compared_{i_snr}.png')


prev_file = ''
temp_input = []
temp_mask = []
temp_input_abs = []
temp_target = []
temp_target_biasf = []
# gen.container_file_info[0]['slice_count']
for i_file in np.arange(gen.__len__()):
    cur_file = gen.container_file_info[0]['file_list'][i_file]
    print(cur_file, end='\r')
    if (cur_file != prev_file) and len(temp_input) > 0:
        file_name, _ = os.path.splitext(prev_file)
        input_stacked = np.stack(temp_input, axis=1)
        # Used to save some memory.. bleh
        input_stacked = harray.scale_minmax(input_stacked, is_complex=True, axis=(0, -2, -1))
        input_stacked_split = np.stack([input_stacked.real, input_stacked.imag], axis=0)
        test = img_as_ubyte(input_stacked_split)
        print('Shape of input stacked ', test.shape)
        # Using NPY because that is easier with my data...
        np.save(os.path.join(ddest_input, f"{file_name}.npy"), test)
        # Store input
        input_abs_stacked = np.concatenate(temp_input_abs)
        print('Shape of abs input stacked ', input_abs_stacked.shape)
        nibabel_obj = nibabel.Nifti1Image(input_abs_stacked.T[::-1, ::-1], np.eye(4))
        nibabel.save(nibabel_obj, os.path.join(ddest_input_abs_sum, f"{file_name}.nii.gz"))
        # Store target
        target_stacked = np.concatenate(temp_target)
        print('Shape of target ', target_stacked.shape)
        nibabel_obj = nibabel.Nifti1Image(target_stacked.T[::-1, ::-1], np.eye(4))
        nibabel.save(nibabel_obj, os.path.join(ddest_target, f"{file_name}.nii.gz"))
        # Store mask
        mask_stacked = np.concatenate(temp_mask)
        print('Shape of mask ', mask_stacked.shape)
        nibabel_obj = nibabel.Nifti1Image(mask_stacked.T[::-1, ::-1], np.eye(4))
        nibabel.save(nibabel_obj, os.path.join(ddest_mask, f"{file_name}.nii.gz"))
        temp_input = []
        temp_mask = []
        temp_input_abs = []
        temp_target = []
    cont = gen.__getitem__(i_file)
    input_array = np.abs(cont['input'].numpy()[::2] + 1j * cont['input'].numpy()[1::2])
    input_abs = np.sqrt(cont['input'].numpy()[::2] ** 2 + cont['input'].numpy()[1::2] ** 2).sum(axis=0)
    input_abs = harray.scale_minmax(input_abs)
    input_abs = img_as_ubyte(input_abs)[None]
    target = cont['target'].numpy()
    target = harray.scale_minmax(target)
    target = img_as_ubyte(target)
    mask = cont['mask'].numpy()
#
# inp_masked = np.ma.masked_where(1-mask[0:1], input_abs)
# fig_obj = hplotc.ListPlot([inp_masked], ax_off=True)
# fig_obj.figure.savefig('/local_scratch/sharreve/input_test.png', bbox_inches='tight', pad_inches=0.0)
# fig_obj = hplotc.ListPlot([target[0]], ax_off=True)
# fig_obj.figure.savefig('/local_scratch/sharreve/input_test.png', bbox_inches='tight', pad_inches=0.0)
    # Store it in a temp array..
    temp_input.append(input_array)
    temp_mask.append(mask)
    temp_input_abs.append(input_abs)
    temp_target.append(target)
    prev_file = cur_file
