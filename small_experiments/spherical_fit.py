"""
After reading and checking everything 100000 times..
"""

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

container = gen.__getitem__(0)
hplotf.plot_3d_list(container['input'][None], cbar=True, vmin=(-0.5, 0.5))
hplotf.plot_3d_list(container['target'][None])
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
sel_array = b1marray_np[sel_coil]

for degree in range(2, 20, 4):
    A_spacy = hspacy.get_spherical_matrix(2 * n_y, n_degree=degree, x_range=(-1, 1), y_range=(-1, 1))
    n_equations = A_spacy.shape[-1]
    A_reshp = A_spacy.reshape((2 * n_y, 2 * n_x, n_equations))

    A_spacy_list = []
    for sel_x, sel_y in coil_pos:
        A_subset = A_reshp[n_x - sel_x: 2 * n_x - sel_x, n_y - sel_y: 2 * n_y - sel_y]
        A_spacy_list.append(A_subset.reshape((-1, n_equations)))

    spherical_matrix = A_spacy_list[sel_coil]
    print('Size matrix ', spherical_matrix.shape)

    A_approx = hspacy.spacy_approximation(spherical_matrix, sel_array.reshape(-1), mask=mask_array.reshape(-1, 1))
    A_approx = A_approx.reshape(n_y, n_x)
    # hplotf.plot_3d_list([A_approx, sel_array], augm='np.real')
    hplotc.ListPlot([A_approx, sel_array], augm='np.real', vmin=(-0.5, 0.5), title=str(degree))

    # Test the fit....
    #
    epsilon_0 = 8.8 * 1e-12
    mu_0 = 4 * np.pi * 1e-7
    omega = 42.58 * 7
    epsilon_r = 50
    sigma = 0.6
    wave_number = np.sqrt(mu_0 * epsilon_0 * epsilon_r * omega ** 2 + 1j * omega * sigma)

    x_range = np.linspace(-1, 1, 2*n_y)
    delta_x = np.diff(x_range)[0]
    A_xx = np.gradient(np.gradient(A_approx, axis=0) * (1/delta_x), axis=0) * (1/delta_x)
    A_yy = np.gradient(np.gradient(A_approx, axis=1) * (1 / delta_x), axis=1) * (1 / delta_x)
    hplotc.ListPlot([A_xx + A_yy + wave_number ** 2 * A_approx], augm='np.real')


