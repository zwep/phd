import tooling.shimming.b1shimming_single as mb1
import scipy.io
import numpy as np
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import data_generator.InhomogRemoval as data_gen
import h5py
import os
import helper.array_transf as harray


"""
Test shimming function...

"""



"""Single experiment with dummy data"""

# Create dummy data
n_size = 100
x_std = 20
blob_list = []
for mu_x in [-5, 5]:
    for mu_y in [-5, 5]:
        x_range = np.linspace(-2*np.pi, 2*np.pi, n_size)
        X, Y = np.meshgrid(x_range, x_range)
        Z = 2 * np.pi * np.exp(-((X-mu_x) ** 2 + (Y-mu_y) ** 2) / x_std * 1j) * np.exp(-((X-mu_x) ** 2 + (Y-mu_y) ** 2) / x_std)
        blob_list.append(Z)

# Create a single mask..
complex_coils = np.array(blob_list)
hplotc.ListPlot(complex_coils, augm='np.angle')
hplotc.ListPlot(complex_coils.sum(axis=0), augm='np.angle')
hplotc.ListPlot(complex_coils.sum(axis=0), augm='np.abs')

mask_obj = hplotc.MaskCreator(complex_coils)
mask = mask_obj.mask

#
shimming_obj = mb1.ShimmingProcedure(complex_coils, mask, amp_shim=False, opt_method='CG',
                                     str_objective='flip_angle', maxiter=5000, debug=True)

for k, i_objective in shimming_obj.objective_dict.items():
    try:
        x_opt, final_value = shimming_obj.find_optimum(objective=i_objective)
        complex_coils_shimmed = np.einsum("tmn, t -> mn", complex_coils, x_opt)
        print(k, final_value)
        hplotc.ListPlot([complex_coils.sum(axis=0), complex_coils_shimmed * (1 + mask)], augm='np.abs', title=k)
    except ValueError:
        print('Error with ', i_objective)
        continue

"""Single experiment with coil data"""

mat_data = '/media/bugger/MyBook/data/simulated/prostate/b1p_b1m_flavio_mat/M16.mat'
mat_obj = scipy.io.loadmat(mat_data)
mask_array = mat_obj['Model']['Mask'][0][0]

b1p_array = np.moveaxis(mat_obj['Model']['B1plus'][0][0], -1, 0)
b1p_array_relative = b1p_array * np.exp(-1j * np.angle(b1p_array[0]))

mask_fraction = 0.05
n_y, n_x = b1p_array_relative.shape[-2:]
y_offset = 0
x_offset = 0

y_center = n_y // 2 + y_offset
x_center = n_x // 2 + x_offset
center_mask = np.zeros((n_y, n_x))
delta_x = int(mask_fraction * n_x)
delta_y = int(mask_fraction * n_y)
center_mask[y_center - delta_y:y_center + delta_y, x_center - delta_x:x_center + delta_x] = 1

import importlib
importlib.reload(mb1)
shimming_obj = mb1.ShimmingProcedure(b1p_array_relative, center_mask, amp_shim=False, opt_method='L-BFGS-B',
                                     relative_phase=True, str_objective='signal_se', maxiter=5000, debug=True,
                                     scale_factor=False)

x_opt, final_value = shimming_obj.find_optimum()
complex_coils_shimmed = harray.apply_shim(b1p_array_relative, cpx_shim=x_opt)
hplotf.plot_3d_list([b1p_array_relative.sum(axis=0), complex_coils_shimmed * (1+center_mask)], augm='np.abs')


"""Single experiment with registered coil data"""

# dir_data = '/media/bugger/MyBook/data/semireal/prostate_simulation_rxtx'
dir_data = '/home/bugger/Documents/data/test_clinic_registration/registrated_h5'
gen = data_gen.DataGeneratorInhomogRemovalH5(ddata=dir_data, dataset_type='test', target_type='biasfield', debug=True, file_ext='h5')
target_dir = gen.container_file_info[0]['target_dir']
sel_file = gen.container_file_info[0]['file_list'][0]
b1_file = os.path.join(target_dir, sel_file)

with h5py.File(b1_file, 'r') as f:
    A = np.array(f['data'])

sel_slice = A.shape[0] // 2
b1_array = A[sel_slice]
b1p_relative = b1_array * np.exp(-1j * np.angle(b1_array[0]))
b1p_relative = harray.correct_mask_value(b1p_relative, mask)

mask = gen.create_random_center_mask(b1_array.shape)

# Needed as a placeholder to get all the objectives....
shimming_obj = mb1.ShimmingProcedure(b1p_relative, mask)

for k, i_objective in shimming_obj.objective_dict.items():
    try:
        shimming_obj.objective = i_objective
        x_opt, final_value = shimming_obj.find_optimum()
        complex_coils_shimmed = harray.apply_shim(b1p_relative, cpx_shim=x_opt)
        print(k, final_value)
        hplotc.ListPlot([shimming_obj.input_array.sum(axis=0), complex_coils_shimmed * (mask + 1)], augm='np.abs', title=k)
    except ValueError:
        print('Error with ', i_objective)
        continue

""" Now do the experiment again, with a fixed optimization string but with different masks.."""

dir_data = '/home/bugger/Documents/data/test_clinic_registration/registrated_h5'
gen = data_gen.DataGeneratorInhomogRemovalH5(ddata=dir_data, dataset_type='test', target_type='biasfield', debug=True, file_ext='h5')
target_dir = gen.container_file_info[0]['target_dir']
sel_file = gen.container_file_info[0]['file_list'][0]
b1_file = os.path.join(target_dir, sel_file)

with h5py.File(b1_file, 'r') as f:
    A = np.array(f['data'])

sel_slice = A.shape[0] // 2
b1_array = A[sel_slice]
b1p_relative = b1_array * np.exp(-1j * np.angle(b1_array[0]))
# b1p_relative = harray.correct_mask_value(b1p_relative, mask)
for _ in range(15):
    mask = gen.create_random_center_mask(b1_array.shape)
    # Needed as a placeholder to get all the objectives....
    shimming_obj = mb1.ShimmingProcedure(b1p_relative, mask, str_objective='b1')
    x_opt, final_value = shimming_obj.find_optimum()
    complex_coils_shimmed = harray.apply_shim(b1p_relative, cpx_shim=x_opt)
    print(final_value)
    hplotc.ListPlot([shimming_obj.input_array.sum(axis=0), complex_coils_shimmed * (mask + 1)], augm='np.abs')

"""
And now with some B1 shim series of cardiac thing..
"""