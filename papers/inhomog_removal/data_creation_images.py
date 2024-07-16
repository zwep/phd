"""
Hier gaan we alle dingen maken die in het paper komen...

Elk plaatje, elk array gaat apart worden opgeslagen.
Ofwel remote... Ofwel lokaal..
"""

import matplotlib.pyplot as plt
import os
import pydicom
import h5py
import pydicom
import scipy.io
import numpy as np
import helper.plot_class as hplotc
import helper.array_transf as harray

remote_location = '/local_scratch/sharreve/paper/inhomog_removal'
local_location = '/home/bugger/Documents/paper/inhomogeneity removal'
local_location_input = os.path.join(local_location, 'data_creation')
local_location_input_per_coil = os.path.join(local_location_input, 'data_per_coil')

# Plaatje 1 [LOCAL]
# Voorbeeld inhomogeneteiten op 7T...

# 7T example
ddata = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/7TMRI005/IM_0431_T2w.dcm'
dicom_obj = pydicom.read_file(ddata)
example_7T = dicom_obj.pixel_array
np.save(os.path.join(local_location_input, 'example_7T'), example_7T)


# 3T example [LOCAL]
ddata = '/media/bugger/MyBook/data/multiT_scan/prostaat_dicom/1_MR/MRI/20201210_0008.h5'
with h5py.File(ddata, 'r') as f:
    n_slices = f['data'].shape[0]
    example_3T = np.array(f['data'][n_slices//2])

np.save(os.path.join(local_location_input, 'example_3T'), example_3T)

# 1.5T example [LOCAL]
ddata = '/media/bugger/MyBook/data/multiT_scan/prostaat_dicom/1_MR/MRL/20201231_0002.h5'
with h5py.File(ddata, 'r') as f:
    n_slices = f['data'].shape[0]
    example_1p5T = np.array(f['data'][n_slices//2-5])

np.save(os.path.join(local_location_input, 'example_1p5T'), example_1p5T)

varray = [[(0, np.max(example_7T) * 0.75), (0, np.max(example_3T) * 1), (0, np.max(example_1p5T) * 1)]]
plot_obj = hplotc.ListPlot([[example_7T, example_3T, example_1p5T]], ax_off=True, vmin=varray)
plot_obj.figure.savefig(os.path.join(local_location_input, 'comparisson_many_T.png'), bbox_inches='tight')

# Showing only the 1.5 and 7T image
varray = [[(0, np.max(example_7T) * 0.75), (0, np.max(example_1p5T) * 1)]]
plot_obj = hplotc.ListPlot([[example_7T, example_1p5T]], ax_off=True, vmin=varray)
plot_obj.figure.savefig(os.path.join(local_location_input, 'comparisson_7_1p5T.png'), bbox_inches='tight')

# Showing only the 7T image
varray = [[(0, np.max(example_7T) * 0.75)]]
plot_obj = hplotc.ListPlot([[example_7T]], ax_off=True, vmin=varray)
plot_obj.figure.savefig(os.path.join(local_location_input, 'example_7T.png'), bbox_inches='tight')

# Plaatje 2 [LOCAL]
# Voorbeeld flavio data (origineel)

# Example B1+ B1-
ddata = '/media/bugger/MyBook/data/simulated/b1p_b1m_flavio_mat/M01.mat'
mat_obj = scipy.io.loadmat(ddata)

# Example B1 Plus
b1_plus_array = np.moveaxis(mat_obj['Model']['B1plus'][0][0], -1, 0)
b1_plus_array = harray.scale_minmax(b1_plus_array, is_complex=True)
np.save(os.path.join(local_location_input, 'example_b1_plus'), b1_plus_array)

plot_obj = hplotc.ListPlot([b1_plus_array], augm='np.abs', ax_off=True, start_square_level=2, cmap='viridis')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.pause(2)
plot_obj.figure.savefig(os.path.join(local_location_input, 'example_b1_plus.png'), bbox_inches='tight')

for i_coil, i_array in enumerate(b1_plus_array):
    plot_obj = hplotc.ListPlot([i_array], augm='np.abs', ax_off=True, cmap='viridis')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.pause(2)
    plot_obj.figure.savefig(os.path.join(local_location_input_per_coil, f'example_b1_plus_coil_{i_coil}.png'), bbox_inches='tight')


# Example B1 Minus
b1_minus_array = np.moveaxis(mat_obj['Model']['B1minus'][0][0], -1, 0)
b1_minus_array = harray.scale_minmax(b1_minus_array, is_complex=True)
np.save(os.path.join(local_location_input, 'example_b1_minus'), b1_minus_array)

plot_obj = hplotc.ListPlot([b1_minus_array], augm='np.abs', ax_off=True, start_square_level=2, cmap='viridis')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.pause(2)
plot_obj.figure.savefig(os.path.join(local_location_input, 'example_b1_minus.png'), bbox_inches='tight')

for i_coil, i_array in enumerate(b1_minus_array):
    plot_obj = hplotc.ListPlot([i_array], augm='np.abs', ax_off=True, cmap='viridis')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.pause(2)
    plot_obj.figure.savefig(os.path.join(local_location_input_per_coil, f'example_b1_minus_coil_{i_coil}.png'), bbox_inches='tight')


# Plaatje 3 [LOCAL]
# Voorbeeld 1.5T data (origineel)
# Prostate data
ddata = '/media/bugger/MyBook/data/multiT_scan/prostaat_dicom/1_MR/MRL/20210105_0003.h5'
with h5py.File(ddata, 'r') as f:
    n_slices = f['data'].shape[0]
    example_1p5T = np.array(f['data'][n_slices // 2 - 5])

plot_obj = hplotc.ListPlot(example_1p5T, ax_off=True)
plot_obj.figure.savefig(os.path.join(local_location_input, 'example_1p5T.png'), bbox_inches='tight')

# Plaatje 4 [REMOTE]
# Maskers
# Mask B1+ B1-
MR_file_name = '1_MR_20210112_0002_transversal'
MR_mask_name = '20210112_0002_transversal'
Flavio_file_name = 'M02'
registrated_name = f"{Flavio_file_name}_to_{MR_file_name}"

ddata = f'/local_scratch/sharreve/flavio_data/{Flavio_file_name}.mat'
mat_obj = scipy.io.loadmat(ddata)
mask_flavio = mat_obj['Model']['Mask'][0][0]
np.save(os.path.join(remote_location, 'mask_flavio'), mask_flavio)

# Mask prostate_mri_mrl data
mask_prostate = f'/local_scratch/sharreve/mri_data/mask_h5/1_MR/MRL/{MR_mask_name}.h5'
with h5py.File(mask_prostate, 'r') as f:
    n_slices = f['data'].shape[0]
    mask_prostate_array = np.array(f['data'][n_slices // 2 - 5])

# Mask registrated data
reg_mask_prostate = f'/local_scratch/sharreve/mri_data/registrated_h5/train/mask/{registrated_name}.h5'
with h5py.File(reg_mask_prostate, 'r') as f:
    n_slices = f['data'].shape[0]
    reg_mask_prostate_array = np.array(f['data'][n_slices // 2 - 5])

# Mask registrated data
reg_b1_mask_prostate = f'/local_scratch/sharreve/mri_data/registrated_h5/train/mask/{registrated_name}_b1.h5'
with h5py.File(reg_b1_mask_prostate, 'r') as f:
    n_slices = f['data'].shape[0]
    reg_b1_mask_prostate_array = np.array(f['data'][n_slices // 2 - 5])

dest_dir = os.path.join(remote_location, 'mask_prostate')
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)


np.save(os.path.join(dest_dir, 'mask_prostate'), mask_prostate_array)

plot_array = [[mask_flavio, mask_prostate_array, reg_b1_mask_prostate_array, reg_mask_prostate_array]]
plot_obj = hplotc.ListPlot(plot_array, ax_off=True,
                           sub_col_row=(2,2))
plot_obj.figure.savefig(os.path.join(dest_dir, 'example_mask_conversion.png'), bbox_inches='tight')

# Plaatje 5 [REMOTE]
# Voorbeeld flavio data (geregistreerd)

# Train input (b1-) 20201231_0002
ddata = f'/local_scratch/sharreve/mri_data/registrated_h5/train/input/{registrated_name}.h5'
with h5py.File(ddata, 'r') as f:
    n_slices = f['data'].shape[0]
    b1_minus_registrated = np.array(f['data'][n_slices//2-5])


dest_dir = os.path.join(remote_location, 'example_b1_minus_registrated')
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)


np.save(os.path.join(dest_dir, 'example_b1_minus_registrated'), b1_minus_registrated)

plot_obj = hplotc.ListPlot([b1_minus_registrated], augm='np.abs', ax_off=True, start_square_level=2, cmap='viridis')
plot_obj.figure.savefig(os.path.join(dest_dir, 'example_b1_minus_registrated.png'), bbox_inches='tight')


for i_coil, i_array in enumerate(b1_minus_registrated):
    plot_obj = hplotc.ListPlot([i_array], augm='np.abs', ax_off=True, cmap='viridis')
    plot_obj.figure.savefig(os.path.join(dest_dir, f'example_b1_minus_reg_coil_{i_coil}.png'), bbox_inches='tight')


# Train target (b1+)
ddata = f'/local_scratch/sharreve/mri_data/registrated_h5/train/target/{registrated_name}.h5'
with h5py.File(ddata, 'r') as f:
    n_slices = f['data'].shape[0]
    b1_plus_registrated = np.array(f['data'][n_slices//2-5])


dest_dir = os.path.join(remote_location, 'example_b1_plus_registrated')
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)


np.save(dest_dir, b1_plus_registrated)

plot_obj = hplotc.ListPlot([b1_plus_registrated], augm='np.abs', ax_off=True, cmap='viridis')
plot_obj.figure.savefig(os.path.join(dest_dir, 'example_b1_plus_registrated.png'), bbox_inches='tight')


for i_coil, i_array in enumerate(b1_plus_registrated):
    plot_obj = hplotc.ListPlot([i_array], augm='np.abs', ax_off=True, cmap='viridis')
    plot_obj.figure.savefig(os.path.join(dest_dir, f'example_b1_plus_reg_coil_{i_coil}.png'), bbox_inches='tight')



# Plaatje 6 [REMOTE]
# Voorbeeld bias field, input data, summed input data
import data_generator.InhomogRemoval as data_gen
dir_data = '/local_scratch/sharreve/mri_data/registrated_h5'

gen = data_gen.DataGeneratorInhomogRemovalH5(ddata=dir_data,
                                             dataset_type='test', complex_type='cartesian',
                                             center_slice=True,
                                             input_shape=(1, 256, 256),
                                             file_ext='h5',
                                             SNR_mode=20,
                                             b1p_scaling=True,
                                             masked=True,
                                             lower_prob=0,
                                             debug=True,
                                             relative_phase=True,
                                             use_tx_shim=True,
                                             objective_shim='b1',
                                             flip_angle=np.pi / 2,
                                             transform_type_target='abs',
                                             # target_type='biasfield_coil',
                                             # target_type='biasfield',
                                             # target_type='b1p_scaled',
                                             target_type='rho',
                                             bins_expansion=10, shuffle=False)


# temp = [(i, x) for i, x in enumerate(gen.container_file_info[0]['file_list']) if f'{registrated_name}' in x]
# i_index, sel_file = temp[0]
cont = gen.__getitem__(0)

# Voorbeeld complexe data
complex_input_array = cont['input'].numpy()[::2] + 1j * cont['input'].numpy()[1::2]


dest_dir = os.path.join(remote_location, 'example_input_array_cpx')
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)


np.save(os.path.join(dest_dir, 'example_input_array_cpx'), complex_input_array)

plot_obj = hplotc.ListPlot([complex_input_array], ax_off=True, augm='np.abs')
plot_obj.figure.savefig(os.path.join(dest_dir, 'example_input.png'), bbox_inches='tight')


for i_coil, i_array in enumerate(complex_input_array):
    plot_obj = hplotc.ListPlot([i_array], augm='np.abs', ax_off=True, start_square_level=2)
    plot_obj.figure.savefig(os.path.join(dest_dir, f'example_input_coil_{i_coil}.png'), bbox_inches='tight')


# Voorbeeld gesommeerde data (1 to 1)
summed_input_array = np.abs(complex_input_array).sum(axis=0)


dest_dir = os.path.join(remote_location, 'example_summed_input')
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)


np.save(os.path.join(dest_dir, 'example_summed_input'), summed_input_array)

plot_obj = hplotc.ListPlot([summed_input_array], ax_off=True, augm='np.abs')
plot_obj.figure.savefig(os.path.join(dest_dir, 'example_summed_input.png'), bbox_inches='tight')

# To do these things.. we need to have biasfield_coil as target thing
if gen.target_type == 'biasfield_coil':
    target_biasfield = cont['target'].numpy()
    for i_coil, i_array in enumerate(target_biasfield):
        plot_obj = hplotc.ListPlot([i_array], augm='np.abs', ax_off=True, start_square_level=2, cmap='viridis')
        plot_obj.figure.savefig(os.path.join(dest_dir, f'example_biasfield_coil_{i_coil}.png'), bbox_inches='tight')

if gen.target_type == 'b1p_scaled':
    target_b1p = cont['target'].numpy()
    temp_mask = cont['mask'].numpy()
    # target_b1p = harray.scale_minmax(target_b1p)
    plot_obj = hplotc.ListPlot(target_b1p, augm='np.abs', ax_off=True, start_square_level=2, cmap='viridis')
    plot_obj.figure.savefig(os.path.join(dest_dir, f'b1p_field.png'), bbox_inches='tight')


if gen.target_type == 'biasfield':
    target_b1p = cont['target'].numpy()
    target_b1p = harray.scale_minmax(target_b1p)
    plot_obj = hplotc.ListPlot(target_b1p, augm='np.abs', ax_off=True, start_square_level=2, cmap='viridis')
    plot_obj.figure.savefig(os.path.join(dest_dir, f'bias_field.png'), bbox_inches='tight')

if gen.target_type == 'rho':
    target_rho = cont['target'].numpy()
    target_rho = harray.scale_minmax(target_rho)
    plot_obj = hplotc.ListPlot(target_rho, augm='np.abs', ax_off=True)
    plot_obj.figure.savefig(os.path.join(dest_dir, f'rho_img.png'), bbox_inches='tight')


# Voorbeeld bias field
target_biasfield = cont['target'].numpy()[0]

dest_dir = os.path.join(remote_location, 'example_biasfield')
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)



np.save(os.path.join(dest_dir, 'example_biasfield.npy'), target_biasfield)

plot_obj = hplotc.ListPlot([target_biasfield], ax_off=True, vmin=(0, 1), cmap='viridis', cbar=True)
plot_obj.figure.savefig(os.path.join(dest_dir, 'example_biasfield.png'), bbox_inches='tight')

bias_field_array = harray.scale_minmax(target_biasfield)
target_expansion, edge_list = harray.split_array_fourier_basis(bias_field_array, n_bins=9, debug=True)

# Create the expansion from the bias field....
np.save(os.path.join(dest_dir, 'example_expansion'), target_expansion)

for i_coil, i_array in enumerate(target_expansion):
    plot_obj = hplotc.ListPlot([i_array], augm='np.abs', ax_off=True, start_square_level=2)
    plot_obj.figure.savefig(os.path.join(dest_dir, f'example_expansion_basis_{i_coil}.png'), bbox_inches='tight')

plot_obj = hplotc.ListPlot([target_expansion], ax_off=True)
plot_obj.figure.savefig(os.path.join(dest_dir, 'example_expansion.png'), bbox_inches='tight')

vmin_array = [(0, np.abs(x).mean() * 20) for x in edge_list]
# vmin_array[-1] = (0, vmin_array[-1][1] * 0.01)
plot_obj = hplotc.ListPlot([np.array(edge_list)], ax_off=True, augm='np.abs', vmin=vmin_array)
plot_obj.figure.savefig(os.path.join(dest_dir, 'example_edge_list_expansion.png'), bbox_inches='tight')

plot_obj = hplotc.ListPlot([(np.array(np.abs(edge_list)) > 0).sum(axis=0)], ax_off=True, augm='np.abs', cbar=True)
plot_obj.figure.savefig(os.path.join(dest_dir, 'example_edge_list_expansion_bin.png'), bbox_inches='tight')

plot_obj = hplotc.ListPlot([[target_expansion.sum(axis=0), target_biasfield, target_biasfield - target_expansion.sum(axis=0)]], ax_off=True, subtitle=[['Expansion', 'Biasfield', 'Difference']], cbar=True, cmap='viridis')
plot_obj.figure.savefig(os.path.join(dest_dir, 'example_expansion_summed.png'), bbox_inches='tight')

hplotc.close_all()

