
import helper_torch.misc as htmisc
import numpy as np
import helper.plot_class as hplotc
import helper.spacy as hspacy
import data_generator.InhomogRemoval as data_gen
import helper.plot_fun as hplotf
import importlib

# Inhomogeneity removal the old-skool way on Gradient Echo data.
dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation_rxtx'
gen = data_gen.DataGeneratorInhomogRemoval(ddata=dir_data, dataset_type='test', complex_type='cartesian',
                                  input_shape=(1, 256, 256),
                                  alternative_input='/home/bugger/Documents/data/celeba',
                                  bias_field=True,
                                  use_tx_shim=True,
                                  b1m_scaling=False,
                                  b1p_scaling=False,
                                  debug=True,
                                  masked=True)

dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation_t1t2_rxtx'
gen = data_gen.DataGeneratorInhomogRemovalT1T2(ddata=dir_data, dataset_type='test', complex_type='cartesian',
                                               input_shape=(1, 256, 256),
                                               alternative_input='/home/bugger/Documents/data/celeba',
                                               bias_field=False,
                                               use_tx_shim=True,
                                               b1m_scaling=False,
                                               b1p_scaling=False,
                                               debug=True,
                                               masked=True)

container = gen.__getitem__(0)
hplotf.plot_3d_list(container['target'][None])
hplotf.plot_3d_list(container['input'][None], cbar=True, vmin=(-0.5, 0.5))
hplotf.plot_3d_list(container['mask'][None])

mask_array = container['mask'].numpy()
if gen.debug:
    b1parray = container['b1p']
    b1marray = container['b1m']
else:
    b1marray = None

coil_pos = htmisc.get_coil_position_torch(np.abs(b1marray))
b1marray_np = b1marray.numpy()
n_c, n_y, n_x = b1marray_np.shape
sel_coil = 0
sel_array = b1marray_np[::2] + 1j * b1marray_np[1::2]
sel_array = np.abs(sel_array.sum(axis=0))
import skimage.transform as sktransf
sel_array = sktransf.resize(sel_array, (64, 64))
hplotf.plot_3d_list(sel_array)
n_y, n_x = sel_array.shape
mask_array = sktransf.resize(sel_array, (64, 64))
# sel_array = b1marray_np[sel_coil]
# sel_array = container['target'].numpy()[0]

result_degree = []
for degree in range(2, 8, 1):
    print(degree)
    A_spacy = hspacy.get_spacy_matrix(2 * n_y, n_degree=degree, x_range=(-1, 1), y_range=(-1, 1))
    n_equations = A_spacy.shape[-1]
    A_spacy_reshp = A_spacy.reshape((2 * n_y, 2 * n_x, n_equations))

    A_spacy_list = []
    for sel_x, sel_y in coil_pos:
        A_subset = A_spacy_reshp[n_x - sel_x: 2 * n_x - sel_x, n_y - sel_y: 2 * n_y - sel_y]
        A_spacy_list.append(A_subset.reshape((-1, n_equations)))

    hplotc.ListPlot([x.reshape(64, 64) for x in A_spacy_list])
    spacy_matrix = np.concatenate(A_spacy_list, axis=1)

    # cylindrical_matrix = A_cyl_list[sel_coil]
    print('Size matrix ', spacy_matrix.shape)
    # spacy_inv = np.linalg.pinv(spacy_matrix)

    A_approx = hspacy.spacy_approximation(spacy_matrix, sel_array.reshape(-1), mask=mask_array.reshape(-1, 1))
    A_approx = A_approx.reshape(n_y, n_x)
    # hplotf.plot_3d_list([A_approx, sel_array], augm='np.real')
    # hplotc.ListPlot([A_approx, sel_array], augm='np.real', vmin=(-0.5, 0.5), title=str(degree))
    result_degree.append([A_approx, sel_array])


hplotc.ListPlot(result_degree, augm='np.real')


"""
Appending content of another file...
"""


"""
Scripts to test out the SPACY fit to self created data, flavio data and registered data...

"""
import re
import os
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
import data_generator.InhomogRemoval as inhom_data_gen
import orthopoly

import helper.plot_fun as hplotf
import helper.array_transf as harray
import helper.misc as hmisc
import helper.plot_class as hplotc
import helper.spacy as hspacy

"""
Create SPACY matrix for specific size (in this case... 256)
"""

import importlib
importlib.reload(hspacy)
n = 256
n_degree = 5
A = hspacy.get_spacy_matrix(n, n_degree=n_degree, offset_y=0, offset_x=0)

new_shape = hmisc.get_square(A.shape[-1])
plt.plot(A[::10000].real, c='k', alpha=0.5)
plt.plot(A[::10000].imag, c='r', alpha=0.5)
plot_array = A.reshape((-1,) + new_shape)

hplotc.SlidingPlot(plot_array)

"""
Fit the data on T2 data
"""

# # T1 T2 data...
dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation_rxtx'
import importlib
gen = inhom_data_gen.DataGeneratorInhomogRemovalT1T2(ddata=dir_data, complex_type='cartesian',
                                                     dataset_type='test',
                                                     bias_field=True,
                                                     use_tx_shim=True,
                                                     b1m_scaling=True,
                                                     b1p_scaling=True,
                                                     debug=True,
                                                     masked=True)

load_file = True
if load_file:
    res_data = gen.__getitem__(0)
    b1_plus_array = res_data['b1p_array']
    b1_minus_array = res_data['b1m']
    b1_plus_scaled = res_data['b1p']
else:
    i_file = gen.file_list[3]
    b1_plus_file = os.path.join(gen.target_dir, i_file)
    b1_minus_file = os.path.join(gen.input_dir, i_file)

    b1_plus_array = np.load(b1_plus_file)
    b1_minus_array = np.load(b1_minus_file)

    n_chan, n_y, n_x = b1_plus_array.shape

    # Apply a random shim setting
    tx_amp = np.ones(n_chan)
    tx_phase = np.random.normal(0, 0.5 * np.sqrt(np.pi), size=n_chan)
    cpx_tx_shim = np.array(
        [r * np.exp(1j * (phi + np.random.normal(0, 0.02 * np.sqrt(np.pi)))) for r, phi in zip(tx_amp, tx_phase)])
    b1_plus_array = np.einsum("tmn, t -> mn", b1_plus_array, cpx_tx_shim)

    y_center, x_center = (n_y // 2, n_x // 2)
    delta_x = int(0.01 * n_y)
    b1_plus_sub = b1_plus_array[:, y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x]
    b1_plus_mean = np.abs(b1_plus_sub.sum(axis=0).mean())

    # Scale to create B1+ data according to T2 weighted data
    b1_plus_abs_scaled = np.abs(np.sin(np.abs(b1_plus_array) / b1_plus_mean * np.pi / 2) ** 3)
    b1_plus_scaled = b1_plus_abs_scaled * np.exp(1j * np.angle(b1_plus_array.sum(axis=0)))


bias_field = np.abs(b1_minus_array * b1_plus_scaled).sum(axis=0)

hplotf.plot_3d_list(b1_plus_scaled[None], augm='np.real')
hplotf.plot_3d_list(b1_minus_array[None], augm='np.real')
hplotf.plot_3d_list(b1_plus_array[None], augm='np.real')
hplotf.plot_3d_list(bias_field[None])

data_to_be_estimated = b1_plus_scaled[None]
# data_to_be_estimated = b1_plus_array
data_to_be_estimated = b1_minus_array

print('Data to be estimated ', data_to_be_estimated.shape)

res_collection = []
for sel_coil in data_to_be_estimated:
    sel_coil = harray.scale_minmax(sel_coil)
    b = sel_coil.ravel()
    proposed_x = np.linalg.lstsq(A.real, b.real, rcond=None)
    proposed_x = proposed_x[0]

    # Reshape to image
    pred_1 = np.matmul(A.real, proposed_x).reshape(n, n)
    true_0 = b.reshape(n, n)
    res_collection.append([pred_1, true_0])

hplotf.plot_3d_list(res_collection, augm='np.real')

"""
Fit it on whole body data (e.g. Flavio's data)
"""


n = 256
n_degree = 20
A = hspacy.get_spacy_matrix(n, n_degree=n_degree, x_range=(-1, 1), y_range=(-1, 1), offset_x=0.0, offset_y=0.0)

dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation_rxtx'
gen_flavio = inhom_data_gen.DataGeneratorInhomogRemoval(ddata=dir_data, complex_type='cartesian',
                                                        dataset_type='test',
                                                        bias_field=True,
                                                        use_tx_shim=True,
                                                        b1m_scaling=True,
                                                        b1p_scaling=True,
                                                        debug=True,
                                                        masked=True)

container = gen_flavio.__getitem__(0)

mask_array = container['mask'].numpy()
target_array = container['target'].numpy()
hplotf.plot_3d_list([mask_array, target_array])

print('Amount of NaNs in A ', A[np.isnan(A)].shape)
b = target_array.ravel()
n = mask_array.shape[0]
# Select only the values IN the mask
A_mask = A[mask_array.ravel() == 1]
b_mask = b[mask_array.ravel() == 1]

proposed_x = np.linalg.lstsq(A_mask, b_mask, rcond=None)
proposed_x = proposed_x[0]

# Reshape to image
temp_result_masked = np.matmul(A_mask, proposed_x)
temp_result = np.zeros(b.ravel().shape, dtype=complex)
temp_result[mask_array.ravel() == 1] = temp_result_masked
pred_1 = temp_result.reshape(n, n)
true_0 = b.reshape(n, n)

hplotf.plot_3d_list([pred_1, true_0], augm='np.real')


"""
Fit with multiple spacy matrices.....
Or try out more why the fit is not working still...
"""

dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation_rxtx'
gen_flavio = inhom_data_gen.DataGeneratorInhomogRemoval(ddata=dir_data, complex_type='cartesian',
                                                        dataset_type='test',
                                                        bias_field=True,
                                                        use_tx_shim=True,
                                                        b1m_scaling=False,
                                                        b1p_scaling=False,
                                                        debug=True,
                                                        masked=True)

container = gen_flavio.__getitem__(0)

input_array = container['input'].numpy()

input_cpx = input_array[::2] + 1j * input_array[1::2]
input_cpx_norm = np.linalg.norm(input_cpx, axis=0)

b_minus = input_cpx/(input_cpx_norm+1e-5)
b_minus[np.isnan(b_minus)] = 0
b_minus = np.abs(b_minus)
b_minus_max = np.max(b_minus, axis=(1,2))
n_pixels = b_minus.shape[-1]

position_coil = []
for i in range(8):
    x_pos, y_pos = map(int, np.where(b_minus[i] == b_minus_max[i]))\
    # Normalize and translate them
    position_coil.append([y_pos/n_pixels-0.5, x_pos/n_pixels-0.5])

    plt.figure()
    plt.imshow(b_minus[i])
    plt.scatter(y_pos, x_pos, c='r')

mask_array = container['mask'].numpy()
target_array = container['target'].numpy()

hplotf.plot_3d_list([mask_array, target_array])

for n_degree in range(2, 12, 2):
    n = 256
    n_degree = 12
    x_range = (-1, 1)
    y_range = (-1, 1)
    debug = True

    offset_list = position_coil

    A = [hspacy.get_spacy_matrix(n, n_degree=n_degree, x_range=x_range, y_range=y_range,
                                 offset_y=offset_y, offset_x=offset_x, debug=debug) for offset_y, offset_x in offset_list]
    if debug:
        A_plot = [x[1] for x in A]
        hplotf.plot_3d_list([x['r'] for x in A_plot])
        hplotf.plot_3d_list(np.array([x['phi'] for x in A_plot])[None])
        hplotf.plot_3d_list([x['theta'] for x in A_plot])
        A = np.concatenate([x[0] for x in A], axis=1)
    else:
        A = np.concatenate(A, axis=1)

    print('Amount of NaNs in A ', A[np.isnan(A)].shape)
    print('Shape of A ', A.shape)
    b = target_array.ravel()
    n = mask_array.shape[0]

    # Select only the values IN the mask
    A_mask = A[mask_array.ravel() == 1]
    b_mask = b[mask_array.ravel() == 1]

    proposed_x = np.linalg.lstsq(A_mask, b_mask, rcond=None)
    proposed_x = proposed_x[0]

    # Reshape to image
    temp_result_masked = np.matmul(A_mask, proposed_x)
    temp_result = np.zeros(b.ravel().shape, dtype=complex)
    temp_result[mask_array.ravel() == 1] = temp_result_masked
    pred_1 = temp_result.reshape(n, n)
    true_0 = b.reshape(n, n)

    hplotf.plot_3d_list([pred_1, true_0], augm='np.real', title=n_degree)
