
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

hplotf.plot_3d_list(b1parray)

res_phase = np.angle(b1marray.numpy()[0] + 1j * b1marray.numpy()[1])
res = b1marray.numpy()[0]

delta_x = 1/256
res_xx = np.gradient(np.gradient(res, axis=0), axis=0) * (1/delta_x) ** 2
res_yy = np.gradient(np.gradient(res, axis=1), axis=1) * (1/delta_x) ** 2
hplotc.ListPlot([res_xx / res + res_yy / res])
hplotf.plot_3d_list(res_phase)

coil_pos = htmisc.get_coil_position_torch(np.abs(b1marray))
b1marray_np = b1marray.numpy()
n_c, n_y, n_x = b1marray_np.shape
sel_coil = 0
sel_array = b1marray_np[sel_coil]
# sel_array = container['target'].numpy()[0] + 1j * container['target'].numpy()[1]
# hplotc.ListPlot(np.arcsin(sel_array ** (1/3)))
# hplotc.ListPlot(sel_array)

result_degree = []
for degree in range(5, 15, 5):
    print(degree)
    A_cyl = hspacy.get_cylindrical_matrix(2 * n_y, n_order=degree, x_range=(-1, 1), y_range=(-1, 1))
    n_equations = A_cyl.shape[-1]
    A_cyl_reshp = A_cyl.reshape((2 * n_y, 2 * n_x, n_equations))

    A_cyl_list = []
    for sel_x, sel_y in coil_pos:
        A_subset = A_cyl_reshp[n_x - sel_x: 2 * n_x - sel_x, n_y - sel_y: 2 * n_y - sel_y]
        A_cyl_list.append(A_subset.reshape((-1, n_equations)))

    # cylindrical_matrix = np.concatenate(A_cyl_list, axis=1)
    cylindrical_matrix = A_cyl_list[sel_coil]
    print('Size matrix ', cylindrical_matrix.shape)

    A_sph = hspacy.get_spherical_matrix(2 * n_y, n_degree=degree, x_range=(-1, 1), y_range=(-1, 1))
    n_equations = A_sph.shape[-1]
    A_sph_reshp = A_sph.reshape((2 * n_y, 2 * n_x, n_equations))

    A_sph_list = []
    for sel_x, sel_y in coil_pos:
        A_subset = A_sph_reshp[n_x - sel_x: 2 * n_x - sel_x, n_y - sel_y: 2 * n_y - sel_y]
        A_sph_list.append(A_subset.reshape((-1, n_equations)))

    spherical_matrix = A_sph_list[sel_coil]
    # spherical_matrix = np.concatenate(A_sph_list, axis=1)
    print('Size matrix ', spherical_matrix.shape)

    spacy_matrix = np.concatenate([spherical_matrix, cylindrical_matrix], axis=1)
    A_approx = hspacy.spacy_approximation(spacy_matrix, sel_array.reshape(-1), mask=mask_array.reshape(-1, 1))
    A_approx = A_approx.reshape(n_y, n_x)
    # hplotf.plot_3d_list([A_approx, sel_array], augm='np.real')
    # hplotc.ListPlot([A_approx, sel_array], augm='np.real', vmin=(-0.5, 0.5), title=str(degree))
    result_degree.append([A_approx, sel_array])


hplotc.ListPlot(result_degree, augm='np.real')
