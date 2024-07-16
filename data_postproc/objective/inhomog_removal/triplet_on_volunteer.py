
import numpy as np
import os
import reconstruction.ReadCpx as read_cpx
import itertools
import multiprocessing as mp
import time

# Quick test..can be removed
import helper.plot_class as hplotc
dir_cpx = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_03/V9_16834/v9_03022021_1644341_5_3_senserefscanclassicV4'
cpc_obj = read_cpx.ReadCpx(dir_cpx)
A = cpc_obj.get_cpx_img()
hplotc.SlidingPlot(A[:, 0])
hplotc.SlidingPlot(np.moveaxis(np.squeeze(A[0, 1]), -1, 0))

"""
Some initial thing to create homog free pictures

using TRIPLET stuff and TIAMO..?
"""

plot_intermediate = False
looped_svd = False
data_path = '/home/bugger/Documents/data/7T/prostate/2020_06_17/ph_10930'
list_files = os.listdir(data_path)
# Select a subset of all the files
load_img_set = 'b1shim'  # 'radwfsminshim'
sel_file_list = [x for x in list_files if load_img_set in x and x.endswith('cpx')]

loaded_img_files = []
for x in sel_file_list:
    file_path = os.path.join(data_path, x)
    A, A_list = read_cpx.read_cpx_img(file_path, sel_loc=[0])
    loaded_img_files.append(np.squeeze(A))

import helper.plot_class as hplotc
import helper.plot_fun as hplotf

if plot_intermediate:
    hplotc.SlidingPlot(loaded_img_files[0])
    hplotf.plot_3d_list(loaded_img_files[0], augm='np.abs', vmin=(0, np.abs(loaded_img_files[0]).max()*0.4))
    hplotf.plot_3d_list([loaded_img_files[0].sum(axis=0), np.abs(loaded_img_files[0]).sum(axis=0)], augm='np.abs')


"""
Below we start with Triplet B1- correction
"""


global_array = loaded_img_files[0]
_, n_c, im_y, im_x = global_array.shape
n_svd = 1
left_svd_array = np.empty((n_c, im_y, im_x), dtype=np.complex)
svd_array = np.empty((im_y, im_x), dtype=np.complex)
right_svd_array = np.empty((n_c, im_y, im_x), dtype=np.complex)


# I could parralelize this...
def calc_svd(args):
    global global_array
    sel_y, sel_x = args
    sel_array = np.take(global_array, sel_x, axis=-1)
    sel_array = np.take(sel_array, sel_y, axis=-1)
    left_x, eig_x, right_x = np.linalg.svd(sel_array, full_matrices=False)
    right_x = right_x.conjugate().T
    return eig_x[0], left_x[:, 0], right_x[:, 0]


N = mp.cpu_count()
cross_prod = list(itertools.product(range(im_y), range(im_x)))
print('Amount of CPUs ', N)
print('Amount of iterations ', im_y * im_x)

t0 = time.time()
with mp.Pool(processes=N) as p:
    results = p.map(calc_svd, list(cross_prod))

eig_list, left_list, right_list = zip(*results)
for i, i_iter in enumerate(cross_prod):
    sel_y, sel_x = i_iter
    left_svd_array[:, sel_y, sel_x] = left_list[i]
    right_svd_array[:, sel_y, sel_x] = right_list[i]
    svd_array[sel_y, sel_x] = eig_list[i]

t1 = time.time()
print('Amount of time ', t1 - t0)
print('Amount of results..', len(results))
print('Example result ', results[0])

fig_right = hplotf.plot_3d_list(right_svd_array, augm='np.abs')
fig_left = hplotf.plot_3d_list(left_svd_array, augm='np.abs')
fig_right.savefig('/home/bugger/Documents/right_example.jpg')
fig_left.savefig('/home/bugger/Documents/left_example.jpg')

if looped_svd:
    left_svd_array = np.empty((n_c, im_y, im_x), dtype=np.complex)
    svd_array = np.empty((im_y, im_x), dtype=np.complex)
    right_svd_array = np.empty((n_c, im_y, im_x), dtype=np.complex)
    t2 = time.time()
    for sel_pixel_y in range(im_y):
        for sel_pixel_x in range(im_x):
            sel_array = np.take(global_array, sel_pixel_x, axis=-1)
            sel_array = np.take(sel_array, sel_pixel_y, axis=-1)
            left_x, eig_x, right_x = np.linalg.svd(sel_array, full_matrices=False)
            right_x = right_x.conjugate().T
            svd_array[sel_pixel_y, sel_pixel_x] = eig_x[0]
            left_svd_array[:, sel_pixel_y, sel_pixel_x] = left_x[:, 0]
            right_svd_array[:, sel_pixel_y, sel_pixel_x] = right_x[:, 0]  # right_x[0, :]

    t3 = time.time()
    print('Amount of time ', t3 - t2)