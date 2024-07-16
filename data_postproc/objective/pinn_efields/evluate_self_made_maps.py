import h5py
import numpy as np
import helper.plot_class as hplotc
import helper.misc as hmisc
import collections
import os
import helper.array_transf as harray
import os
import objective.pinn_efields.executor_pinn_efields as executor
import objective.pinn_efields.postproc_pinn_efields as postproc


"""
We just need some more maps.. with different shapes..

Lets get some conductivity and permitivity values from Flavios stuff..

"""


ddata = '/home/bugger/Documents/data/pinn_fdtd/Phantom_1 (Dipole 0deg).mat'
ddest_me = '/home/bugger/Documents/data/pinn_fdtd/self_made_sigma_eps/me'
ddest_thierry = '/home/bugger/Documents/data/pinn_fdtd/self_made_sigma_eps/thierry'
mat_obj = hmisc.load_array(ddata)

reshape_size = tuple(mat_obj['Grid_size'][0][::-1])


sel_slice = 25
#
sigma = mat_obj['sigma'].reshape(reshape_size)
sel_sigma = sigma[sel_slice]
sel_sigma = np.round(sel_sigma, 2)
eps = mat_obj['eps'].reshape(reshape_size)
sel_eps = eps[sel_slice]
sel_eps = np.round(sel_eps, 2)
eps_mask = harray.get_treshold_label_mask(eps[25])
#
collections.Counter(sel_sigma.ravel())
collections.Counter(sel_eps.ravel())
#
# hplotc.SlidingPlot(sigma)
# hplotc.ListPlot([sel_eps, sel_sigma, eps_mask])
#
tissue_type_combo = list(set(zip(sel_eps[eps_mask == 1], sel_sigma[eps_mask == 1])))


import helper.plot_class as hplotc
A = hplotc.MaskCreator(np.ones((256, 256)), different_area=True)
extracted_mask = A.mask
unique_mask_values = collections.Counter(extracted_mask.ravel())
# Dont assign the background...
del unique_mask_values[0]
sigma_array = np.zeros(extracted_mask.shape)
eps_array = np.zeros(extracted_mask.shape)
for i_key in unique_mask_values.keys():
    tissue_index = np.random.choice(range(len(tissue_type_combo)))
    temp_eps, temp_sigma = tissue_type_combo[tissue_index]
    sigma_array[extracted_mask == i_key] = temp_sigma
    eps_array[extracted_mask == i_key] = temp_eps

hplotc.ListPlot([eps_array, sigma_array])
ddest_sigma = os.path.join(ddest_me, 'sigma_array.npy')
ddest_eps = os.path.join(ddest_me, 'eps_array.npy')
# np.save(ddest_sigma, sigma_array)
# np.save(ddest_eps, eps_array)

# Convert THierries Data to npy...
ddata = '/home/bugger/Documents/data/pinn_fdtd/thierry/EP_maps'
deps = os.path.join(ddata, 'eps_GT.mat')
dsigma = os.path.join(ddata, 'sig_GT.mat')

eps_array = hmisc.load_array(deps, data_key='eps_GT').T
sigma_array = hmisc.load_array(dsigma, data_key='sig_GT').T
n_slice, _, _ = eps_array.shape
import skimage.transform as sktransf
eps_array_resize = sktransf.resize(eps_array[n_slice//2], (256, 256), anti_aliasing=True, order=0)
sig_array_resize = sktransf.resize(sigma_array[n_slice//2], (256, 256), anti_aliasing=True, order=0)
sel_slice = eps_array.shape
hplotc.ListPlot([sig_array_resize, sigma_array[n_slice//2]])
ddest_sigma = os.path.join(ddata, 'sigma_array.npy')
ddest_eps = os.path.join(ddata, 'eps_array.npy')
np.save(ddest_sigma, sig_array_resize)
np.save(ddest_eps, eps_array_resize)
"""
We created some data...
lets use it as input now.. REMOTELY
Because going local was too much trouble
"""
ddata = '/local_scratch/sharreve/mri_data/self_made_sigma_eps/me'
ddest_sigma = os.path.join(ddata, 'sigma_array.npy')
ddest_eps = os.path.join(ddata, 'eps_array.npy')

ddata = '/local_scratch/sharreve/mri_data/self_made_sigma_eps/thierry'
ddest_sigma = os.path.join(ddata, 'sigma_array.npy')
ddest_eps = os.path.join(ddata, 'eps_array.npy')


sigma_array = np.load(ddest_sigma)
eps_array = np.load(ddest_eps)
mask_array = sigma_array != 0
zero_index = np.argwhere(sigma_array==0)

# Now load a model...
# ddata = '/local_scratch/sharreve/mri_data/pinn_fdtd'
bfield_model_path = f'/local_scratch/sharreve/model_run/pinn_fdtd_b_field_continued/config_00'
efield_model_path = f'/local_scratch/sharreve/model_run/pinn_fdtd_e_field/config_00'
ddest = f'/local_scratch/sharreve'

postproc_obj_B_field = postproc.PostProcPinnEfields(executor_module=executor, ddest=ddest,
                                            config_name='config_param.json',
                                            config_path=bfield_model_path,
                                            debug=True)
postproc_obj_E_field = postproc.PostProcPinnEfields(executor_module=executor, ddest=ddest,
                                            config_name='config_param.json',
                                            config_path=efield_model_path,
                                            debug=True)

import torch
# Create random coil location..
for counter in range(10):
    sel_index = np.random.choice(range(len(zero_index)))
    coil_coord = zero_index[sel_index]
    coil_location = np.zeros(sigma_array.shape)
    coil_location[coil_coord[0], coil_coord[1]] = 1
    input_array = np.stack([eps_array, sigma_array, coil_location])
    X = torch.from_numpy(input_array).float()[None]
    X = X.to(postproc_obj_B_field.modelrun_obj.device)
    with torch.no_grad():
        Bfield_result = postproc_obj_B_field.model_obj(X)
        Efield_result = postproc_obj_E_field.model_obj(X)
    Bfield_array = Bfield_result.cpu().numpy()[0]
    B_x = Bfield_array[0] + 1j * Bfield_array[1]
    B_y = Bfield_array[2] + 1j * Bfield_array[3]
    B_abs = np.stack([np.abs(B_x), np.abs(B_y)])
    B_angle = np.stack([np.angle(B_x), np.angle(B_y)])
    Efield_array = Efield_result.cpu().numpy()[0]
    E_z = Efield_array[0] + 1j * Efield_array[1]
    E_abs_angle = np.stack([np.abs(E_z), np.angle(E_z)])
    fig_obj = hplotc.ListPlot([input_array, B_abs * mask_array, B_angle * mask_array, E_abs_angle * mask_array], figsize=(20, 20),
                              cbar=True, subtitle=[['Eps', 'Sigma', 'Coil'], ['|B_x|', '|B_y|'], ['∠B_x', '∠B_y'], ['|E_z|', '∠E_z']])
    fig_obj.ax_list[2].scatter(coil_coord[1], coil_coord[0])
    fig_obj.figure.savefig(os.path.join(postproc_obj_B_field.ddest, f'example_data_dummy_{counter}.png'))
    hplotc.close_all()