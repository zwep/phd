import numpy as np
import os
import helper.plot_class as hplotc
import helper.misc as hmisc
import h5py
from objective_helper.reconstruction import freebreathing2direct_array, resize_array, convert_direct2cpx_img

"""
We used the yarra server to convert a single free breathing file to .mat
Here we convert that .mat to .npy in DIRECT format
"""

ddata = '/local_scratch/sharreve/mri_data/cardiac_free_breathing/input_mat/'
ddest = '/local_scratch/sharreve/mri_data/cardiac_free_breathing/input/'
# mat_file = 'ca_28022022_1957320_15_1_transverse_radial_retrospective_free_breathingV4.lab.mat'
mat_file = 'v9_05032022_1228321_16_1_transverse_radial_retrospective_free_breathingV4.lab.mat'
dest_file_name = 'free_breathing'

mat_path = os.path.join(ddata, mat_file)


# Shape is
# (30, 10, 24, 1, 400, 400)
sel_card = 15
sel_dyn = 5
with h5py.File(mat_path, 'r') as f:
    array_shape = f['Data'].shape

n_card, n_dyn = array_shape[:2]

for i_card in range(n_card):
    for j_dyn in range(n_dyn):
        print(f'Card {i_card} Dyn {j_dyn}', end='\r')
        dest_path = os.path.join(ddest, dest_file_name + f'_{i_card}_{j_dyn}.h5')
        with h5py.File(mat_path, 'r') as f:
            card_array = np.array(f['Data'][i_card, j_dyn])
            # card_array = f['Data'][i_card, j_dyn]
            card_cpx = np.squeeze(card_array['real'] + 1j * card_array['imag'])
            # Okay.. now moving on to
            # ANd going towards.. ny, nx, ncoil, n_loc = x.shape
            # First.. lets fix ONE card/dyn combination
            img_space_cpx = np.fft.fftshift(np.fft.ifft2(card_cpx), axes=(-2, -1))
            # fig_obj = hplotc.ListPlot(img_space_cpx)
            # fig_obj.savefig('orig_img')
            #
            # fig_obj = hplotc.ListPlot(np.fft.fft2(img_space_cpx))
            # fig_obj.savefig('orig_kspace')
            # ny, nx, ncoil, n_loc = x.shape
            img_space_cpx_prep = np.moveaxis(img_space_cpx, 0, -1)[..., None]
            #
            # I think I need to re-interpolate some stuff..
            direct_array = freebreathing2direct_array(img_space_cpx_prep)
            # direct_array = resize_array(direct_array)
            # Check if direct array functions normal
            # rec_img_cpx = convert_direct2cpx_img(direct_array)
            # fig_obj = hplotc.ListPlot(np.sqrt((np.abs(rec_img_cpx)**2).sum(axis=0)))
            # fig_obj.savefig('rec_img')
            #
            # direct_array.append(temp_array)
        #
        direct_array = np.array(direct_array).astype(np.float32)
        with h5py.File(dest_path, 'w') as f:
            f.create_dataset('kspace', data=direct_array)

