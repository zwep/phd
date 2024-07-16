
"""
We have succesfully used the radial data..

Now we want to link the cartesian data to this radial data...

Here we might use a model... or some other/better registration process
"""

import pandas as pd
import skimage.transform as sktransf
import time
import reconstruction.ReadCpx as read_cpx
import scipy.io
import matplotlib.pyplot as plt
import scipy.io
import os
import numpy as np
import helper.plot_class as hplotc
import re
import itertools

# Search for high-time stuff first
ddata_scan = '/media/bugger/MyBook/data/7T_scan/cardiac'
ddata_cartesian_unfolded = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac'
re_identifier = re.compile("([a-z0-9]{2}_[0-9]*_[0-9]*)_[0-9]*.*")
target_dir = '/media/bugger/MyBook/data/7T_data'

data_type = 'train'
plt.figure(num=1)

counter = 0
overview_list = []
for d, _, f in os.walk(ddata_scan):
    filter_f = [x for x in f if 'high_time' in x and x.endswith('cpx')]
    if len(filter_f):
        print(counter, d)
        if counter > 12:
            data_type = 'validation'
        if counter > 14:
            data_type = 'test'

        counter += 1
        for ifile in filter_f:
            if 'transradialfast' in ifile:
                cartesian_name = 'cine1slicer'
                subdir = 'cartesian_radial_transverse'
            elif 'p2ch' in ifile:
                cartesian_name = 'p2chV4'
                subdir = 'cartesian_radial_p2ch'
            elif '4ch' in ifile:
                cartesian_name = '4chV4'
                subdir = 'cartesian_radial_4ch'
            else:
                subdir = 'misc'
                cartesian_name = None
                print('Eeeh')

            found_identifier = [re_identifier.findall(x) for x in f if cartesian_name in x]
            single_identifier = list(set(itertools.chain(*found_identifier)))[0]

            for d1, _, f1 in os.walk(ddata_cartesian_unfolded):
                filter_f1 = [x for x in f1 if single_identifier in x and (x.endswith('.npy') or x.endswith('.mat'))]
                if len(filter_f1):
                    temp_dict = {}
                    target_sub_dir = os.path.join(target_dir, subdir)

                    radial_high_time_path = os.path.join(d, ifile)
                    cartesian_path = os.path.join(d1, filter_f1[0])
                    print('\t', data_type)
                    print('\t\t', radial_high_time_path)
                    print('\t\t', cartesian_path)
                    temp_dict['data_type'] = data_type
                    temp_dict['cartesian_file'] = cartesian_path
                    temp_dict['radial_file'] = radial_high_time_path

                    # Read cartesian file
                    cpx_obj = read_cpx.ReadCpx(radial_high_time_path)
                    cpx_array = cpx_obj.get_cpx_img()
                    cpx_array = np.squeeze(cpx_array)
                    n_card_radial = cpx_array.shape[1]

                    if cartesian_path.endswith('mat'):
                        cart_array = scipy.io.loadmat(cartesian_path)['reconstructed_data']
                        cart_array = np.moveaxis(np.squeeze(cart_array), -1, 0)
                    else:
                        cart_array = np.squeeze(np.load(cartesian_path))

                    # Plot a single image to fix the orientation...
                    single_radial_image = np.abs(cpx_array.sum(axis=0)[n_card_radial//2])
                    single_cart_image = np.abs(cart_array[n_card_radial // 2])
                    hplotc.ListPlot([single_cart_image, np.array([single_radial_image, single_radial_image[::-1, :], single_radial_image[:, ::-1],
                                     single_radial_image[::-1, ::-1]])], start_square_level=1, ax_off=True, fignum=1,
                                     subtitle=[[-1], [1, 2, 3, 4]])
                    plt.pause(1.5)
                    x = input()
                    x = int(x)
                    if x == 1:
                        final_radial_image = cpx_array
                    elif x == 2:
                        final_radial_image = cpx_array[:, :, ::-1]
                    elif x == 3:
                        final_radial_image = cpx_array[:, :, :, ::-1]
                    elif x == 4:
                        final_radial_image = cpx_array[:, :, ::-1, ::-1]

                    temp_dict['orientation'] = x
                    print('Orientation', x)
                    overview_list.append(temp_dict)

                    # Now store the new orientation and the radial file...
                    dest_radial_image_dir = os.path.join(target_sub_dir, data_type, 'input')
                    dest_cart_image_dir = os.path.join(target_sub_dir,  data_type, 'target')
                    if not os.path.isdir(dest_radial_image_dir):
                        os.makedirs(dest_radial_image_dir)

                    if not os.path.isdir(dest_cart_image_dir):
                        os.makedirs(dest_cart_image_dir)

                    dest_radial_image_path = os.path.join(dest_radial_image_dir, single_identifier)
                    dest_cart_image_path = os.path.join(dest_cart_image_dir, single_identifier)
                    temp_dict['target_radial'] = dest_radial_image_path
                    temp_dict['target_cartesian'] = dest_cart_image_path
                    # Will do this with the created .csv.. because otherwise it takes too long.
                    # print('Saving images...')
                    # np.save(dest_radial_image_path, final_radial_image)
                    # np.save(dest_cart_image_path, cart_array)
                    # print('Done')

#
pd_dataframe = pd.DataFrame.from_dict(overview_list)
csv_path = os.path.join(target_dir, 'orientation_radial_cartesian.csv')
pd_dataframe.to_csv(csv_path, index=False)


csv_path = os.path.join(target_dir, 'orientation_radial_cartesian.csv')
## Now perform the actual copy-ing....

A = pd.read_csv(csv_path)
for i_index, irow in A.iterrows():
    print(i_index)
    if i_index < 23:
        continue
    t0 = time.time()
    id_file = os.path.basename(irow.target_radial)
    orientation = irow.orientation
    base_path = os.path.dirname(os.path.dirname(irow.target_radial))
    data_type = irow.data_type
    dest_input_file = os.path.join(base_path, data_type, 'input', id_file)
    dest_target_file = os.path.join(base_path, data_type, 'target', id_file)

    cartesian_path = irow.cartesian_file
    radial_high_time_path = irow.radial_file

    if cartesian_path.endswith('mat'):
        cart_array = scipy.io.loadmat(cartesian_path)['reconstructed_data']
        cart_array = np.moveaxis(np.squeeze(cart_array), -1, 0)
    else:
        cart_array = np.squeeze(np.load(cartesian_path))

    # Read cartesian file
    cpx_obj = read_cpx.ReadCpx(radial_high_time_path)
    cpx_array = cpx_obj.get_cpx_img()
    cpx_array = np.squeeze(cpx_array)
    n_card_radial = cpx_array.shape[1]

    if orientation == 1:
        final_radial_image = cpx_array
    elif orientation == 2:
        final_radial_image = cpx_array[:, :, ::-1]
    elif orientation == 3:
        final_radial_image = cpx_array[:, :, :, ::-1]
    elif orientation == 4:
        final_radial_image = cpx_array[:, :, ::-1, ::-1]

    # Reshape already..
    nc, ncard = final_radial_image.shape[:2]
    import helper.array_transf as harray
    # Simply back tp 256
    new_shape = (nc, ncard, 256, 256)
    print('Resizing images...', time.time() - t0)
    final_radial_image_resize = harray.resize_complex_array(final_radial_image, new_shape=(new_shape), preserve_range=True)

    ncard_cart = cart_array.shape[0]
    new_shape_cart = (ncard_cart, 256, 256)
    cart_array_resize = harray.resize_complex_array(cart_array, new_shape=(new_shape_cart), preserve_range=True)

    print('Saving images...', time.time() - t0)
    np.save(dest_input_file, final_radial_image_resize)
    np.save(dest_target_file, cart_array_resize)
    print('Done. ', time.time() - t0)
