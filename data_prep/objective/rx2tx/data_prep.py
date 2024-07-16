# encoding: utf-8

import numpy as np
import os
import helper.misc as hmisc
import re
import getpass
import scipy.signal
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma

"""
Here we do certain prep-data steps
 Other steps are shown in Deep Learning in MRI project

"""

check_phase_stuff = False  # Piece of code later in the file
create_masked = True

if getpass.getuser() == 'bugger':
    orig_path = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels'
    if create_masked:
        dest_path = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels_svd_masked'
    else:
        dest_path = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels_svd'
else:
    orig_path = '/home/seb/data/b1shimsurv_all_channels'
    dest_path = '/home/seb/data/b1shimsurv_all_channels_svd'

hmisc.create_datagen_dir(dest_path, data_list=('input', 'target', 'svd'))

for d_dir, _, f_files in os.walk(orig_path):
    print(d_dir)
    counter = 0
    if os.path.basename(d_dir) == 'input':
        f_files = [x for x in f_files if x.endswith('.npy')]
        n_files = len(f_files)
        for i, i_file in enumerate(f_files):
            if i % (int(n_files * 0.05)+1) == 0:
                print(int(100 * i/n_files), '%')

            file_name, file_ext = os.path.splitext(i_file)
            file_path = os.path.join(d_dir, i_file)
            # Create new paths
            dest_path_input = re.sub('b1shimsurv_all_channels', os.path.basename(dest_path), d_dir)
            dest_path_target = re.sub('input', 'target', dest_path_input)
            dest_path_svd = re.sub('input', 'svd', dest_path_input)

            # Create new file names
            dest_file_left = os.path.join(dest_path_input, file_name + "_left" + file_ext)
            dest_file_right = os.path.join(dest_path_target, file_name + "_right" + file_ext)
            dest_file_svd = os.path.join(dest_path_svd, file_name + "_svd" + file_ext)

            if os.path.isfile(dest_file_left) and os.path.isfile(dest_file_right) and os.path.isfile(dest_file_svd):
                print('We have already created file ', file_name, counter)
                counter += 1
                continue
            else:
                print('We are loading file ', file_name)

            input_array = np.load(file_path)
            # We either use a mask.. or do noise minimization.
            if create_masked:
                c_summed = np.abs(input_array).sum(axis=0).sum(axis=0)
                treshhold = np.max(c_summed) * 0.1
                c_tresh = (c_summed > treshhold).astype(int)
                n_mask = 32
                kernel = np.ones((n_mask, n_mask)) / n_mask ** 2
                tresh_smooth = scipy.signal.convolve2d(c_tresh, kernel, mode='same', boundary='symm')
                input_array = input_array * tresh_smooth
            else:
                res_real = denoise_tv_chambolle(input_array.real, weight=1.5)
                res_imag = denoise_tv_chambolle(input_array.imag, weight=1.5)
                input_array = res_real + 1j * res_imag

            n_c, n_c, im_y, im_x = input_array.shape
            n_svd = 1
            left_svd_array = np.empty((n_c, im_y, im_x), dtype=np.complex)
            svd_array = np.empty((im_y, im_x), dtype=np.complex)
            right_svd_array = np.empty((n_c, im_y, im_x), dtype=np.complex)

            for sel_pixel_y in range(im_y):
                for sel_pixel_x in range(im_x):
                    sel_array = np.take(input_array, sel_pixel_x, axis=-1)
                    sel_array = np.take(sel_array, sel_pixel_y, axis=-1)
                    left_x, eig_x, right_x = np.linalg.svd(sel_array, full_matrices=False)
                    right_x = right_x.conjugate().T
                    svd_array[sel_pixel_y, sel_pixel_x] = eig_x[0]
                    left_svd_array[:, sel_pixel_y, sel_pixel_x] = left_x[:, 0]
                    right_svd_array[:, sel_pixel_y, sel_pixel_x] = right_x[:, 0]  # right_x[0, :]

            np.save(dest_file_left, left_svd_array)
            np.save(dest_file_right, right_svd_array)
            np.save(dest_file_svd, svd_array)
            print('Stored file ', file_name)


# Check relative phase stuff
if check_phase_stuff:
    import helper.plot_class as hplotc
    import helper.plot_fun5 as hplotf

    for d_dir, _, f_files in os.walk(orig_path):
        print(d_dir)
        if os.path.basename(d_dir) == 'input':
            f_files = [x for x in f_files if x.endswith('.npy')]
            n_files = len(f_files)

            for i, i_file in enumerate(f_files):
                if i % (int(n_files * 0.05)+1) == 0:
                    print(int(100 * i/n_files), '%')

                file_name, file_ext = os.path.splitext(i_file)
                file_path = os.path.join(d_dir, i_file)

                input_array = np.load(file_path)

                hplotc.SlidingPlot(input_array)
                hplotf.plot_3d_list(input_array, augm='np.angle')
                hplotf.plot_3d_list(input_array[:, 0], augm='np.angle')
                phi_rel = np.angle(input_array[:, 0])
                complex_phi = np.exp(-1j * phi_rel)
                rel_input_array = input_array * complex_phi[:, np.newaxis]
                hplotf.plot_3d_list(rel_input_array, augm='np.angle')
                hplotf.plot_3d_list(rel_input_array, augm='np.abs')
                hplotf.plot_3d_list(rel_input_array, augm='np.real')
                hplotf.plot_3d_list(rel_input_array, augm='np.imag')