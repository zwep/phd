"""
Here we will test the functionality of finding the coil position.. and test for spacy fit
"""

import helper.plot_fun as hplotf
import numpy as np
import torch
import helper_torch.misc as htmisc
import helper.spacy as hspacy
import helper.plot_class as hplotc
import data_generator.InhomogRemoval as data_gen
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

inp_np = container['input']
tgt_np = container['target']
msk_np = container['mask']
if gen.debug:
    b1parray = container['b1p']
    b1marray = container['b1m']

n_pixels = inp_np.shape[-1]
n2_pixels = n_pixels * 2
# Now get a spacy matrix that is quite big...

import helper.misc as hmisc

coil_pos = htmisc.get_coil_position_torch(np.abs(b1marray))
A_spacy = hspacy.get_spacy_matrix(n2_pixels, n_degree=6, x_range=(-2, 2), y_range=(-2, 2))
n_equations = A_spacy.shape[-1]

A_reshp = A_spacy.reshape((n2_pixels, n2_pixels, n_equations))

A_spacy_list = []
for sel_x, sel_y in coil_pos:
    A_subset = A_reshp[n_pixels - sel_x: 2 * n_pixels - sel_x,
               n_pixels - sel_y: 2 * n_pixels - sel_y]
    A_spacy_list.append(A_subset.reshape((-1, n_equations)))

A_spacy_coils = np.concatenate(A_spacy_list, axis=1)
A_spacy_tens = torch.from_numpy(A_spacy_coils.real).float()

# Visually inspect stuff
A_spacy_img = [x.reshape((n_pixels, n_pixels, n_equations)) for x in A_spacy_list]
hplotc.SlidingPlot(np.moveaxis(np.stack(A_spacy_img, axis=0), -1, 0))

# Approximate something with Torch with and without mask
approx_thing = torch.sqrt(b1parray[0] ** 2 + b1parray[1] ** 2)
approx_thing = tgt_np[0]
approx_thing = b1parray.numpy()[0] + 1j * b1parray.numpy()[1]
approx_thing = b1marray[1].numpy()
approx_thing = b1parray[0]

b_approx = hspacy.spacy_approximation_torch(A_spacy_tens, target_tensor=approx_thing.reshape(-1, ))
b_img = b_approx.reshape(approx_thing.shape[-2:])
hplotf.plot_3d_list([approx_thing.numpy(), b_img.numpy()])

b_approx_mask = hspacy.spacy_approximation_torch(A_spacy_tens, target_tensor=approx_thing.reshape(-1, ),
                                                 mask=msk_np.reshape(-1, 1))
b_img_mask = b_approx_mask.reshape(approx_thing.shape[-2:])
hplotf.plot_3d_list([approx_thing.numpy(), b_img_mask.numpy()])

