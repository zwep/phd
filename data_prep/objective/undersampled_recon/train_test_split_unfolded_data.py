
"""
All data is in a different format..super annoying..

Scale it to 256
"""

import numpy as np
import os
import skimage.transform as sktransf
import reconstruction.ReadCpx as read_cpx
import helper.plot_class as hplotc

dtransverse = '/media/bugger/MyBook/data/7T_scan'
trans_dict = {'filter_on': 'transradialfastV4', 'folder_name_appendix': 'transverse'}
two_ch_dict = {'filter_on': 'p2ch_radialV4', 'folder_name_appendix': 'p2ch'}
four_ch_dict = {'filter_on': '4ch_radialV4', 'folder_name_appendix': '4ch'}
sa_dict = {'filter_on': 'sa_radialV4', 'folder_name_appendix': 'sa'}
target_size = (256, 256)

for temp_dict in [trans_dict, two_ch_dict, four_ch_dict, sa_dict]:
    folder_name_appendix = temp_dict['folder_name_appendix']
    filter_on = temp_dict['filter_on']
    orig_dir = f'/media/bugger/MyBook/data/7T_data/cardiac_radial_{folder_name_appendix}'
    dest_dir = f'/media/bugger/MyBook/data/7T_data/radial_dataset_{folder_name_appendix}'
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    print('\n\n Original dir', orig_dir)
    file_list = [x for x in os.listdir(orig_dir) if x.endswith('npy')]
    for i_file in file_list:
        file_path = os.path.join(orig_dir, i_file)
        dest_path = os.path.join(dest_dir, i_file)

        if not os.path.isfile(dest_path):
            print('File ', i_file)
            card_array = np.load(file_path)
            if card_array.ndim == 3:
                card_array = np.expand_dims(card_array, axis=1)

            print('Shape ', card_array.shape, '-> (256, 256)')

            n_card, n_chan = card_array.shape[:2]

            # This resizing is kinda stupid. Can be done als preprocessing step...
            # Is this echt wat je wilt....?
            temp_real = sktransf.resize(card_array.real, output_shape=(n_card, n_chan, ) + target_size, order=2)
            temp_imag = sktransf.resize(card_array.imag, output_shape=(n_card, n_chan, ) + target_size, order=2)
            temp_cpx = temp_real + 1j * temp_imag

            np.save(dest_path, temp_cpx)
        else:
            print('\t skip file.. ', i_file)