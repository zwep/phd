"""
Well.. weve unfolded a lot of data.
Lets sort it all nicely and such in this file///

- transverse
- P2ch
- 4ch

"""

import helper.misc as hmisc
import numpy as np
import os
import reconstruction.ReadCpx as read_cpx
import helper.plot_class as hplotc

dscan = '/media/bugger/MyBook/data/7T_scan'
trans_dict = {'filter_on': 'transradialfastV4', 'folder_name_appendix': 'transverse'}
two_ch_dict = {'filter_on': 'p2ch_radialV4', 'folder_name_appendix': 'p2ch'}
four_ch_dict = {'filter_on': '4ch_radialV4', 'folder_name_appendix': '4ch'}
sa_dict = {'filter_on': 'sa_radialV4', 'folder_name_appendix': 'sa'}


for temp_dict in [trans_dict, two_ch_dict, four_ch_dict, sa_dict]:
    folder_name_appendix = temp_dict['folder_name_appendix']
    filter_on = temp_dict['filter_on']
    dest_dir = f'/media/bugger/MyBook/data/7T_data/cardiac_radial_{folder_name_appendix}'
    print('\n\n Target dir', dest_dir)
    dest_files = os.listdir(dest_dir)
    dest_basename_files = [hmisc.get_base_name(x) for x in dest_files]
    current_number_of_files = len(dest_files)

    radial_images = []
    for d, _, f in os.walk(dscan):
        filter_f = [x for x in f if x.endswith('cpx') and filter_on in x]
        if len(filter_f) and ('v9' in d.lower() or 'v8' in d.lower()):
            for i_file in filter_f:
                temp_path = os.path.join(d, i_file)
                radial_images.append(temp_path)

    print('Number of radial images \t', len(radial_images), '/', current_number_of_files)
    for i_file in radial_images:
        file_name_no_ext = os.path.splitext(os.path.basename(i_file))[0]
        dest_file = os.path.join(dest_dir, file_name_no_ext)
        # Skip the file if we have already stored it...
        # This makes sure that we also properly check the short axis files
        if any([file_name_no_ext in x for x in dest_basename_files]):
            print('Skipping ', i_file)
            continue
        # Else, read in the .cpx data and get the image
        cpx_obj = read_cpx.ReadCpx(i_file)
        try:
            img_array = cpx_obj.get_cpx_img()
        except TypeError:
            print('Could not process ', i_file)
            continue

        xy_shape = img_array.shape[-2:]
        print(i_file, img_array.shape)

        # Then store the image (per location) to a npy file
        if folder_name_appendix == 'sa':
            n_loc = img_array.shape[1]
            for i_loc in range(n_loc):
                # This is done so that the dimensions are aligned with the other acquisitions...(which have nloc=1)
                dest_file = os.path.join(dest_dir, file_name_no_ext + f'_{str(i_loc).zfill(2)}')
                sel_array = np.squeeze(img_array[:, i_loc])
                np.save(dest_file, sel_array)
        else:
            img_array = np.squeeze(img_array)
            np.save(dest_file, img_array)