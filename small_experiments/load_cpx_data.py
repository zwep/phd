
"""
Check scanned data..
"""
import helper.array_transf as harray
import reconstruction.ReadCpx as read_scan
import os
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import numpy as np
import skimage.transform as sktransf

scan_dir = '/media/bugger/Storage/2020_09_08/se_12359'
scan_dir = '/media/bugger/Storage/2020_09_30/V9_13026'
scan_dir = '/media/bugger/MyBook/data/7T_scan/cardiac/2020_09_30/V9_13026'
scan_dir = '/media/bugger/MyBook/data/7T_scan/prostate_mri_mrl/2021_01_06/pr_16289'
scan_dir = '/media/bugger/MyBook/data/7T_scan/prostate_mri_mrl/2021_01_06/pr_16289'

# scan_dir = '/media/bugger/Storage/2020_11_04/V9_13975'
file_list = [x for x in os.listdir(scan_dir) if x.endswith('cpx')]

cine_file = [x for x in file_list if 'cine' in x]
sense_ref_file = [x for x in file_list if 'sense' in x]

cine_scan, _ = read_scan.read_cpx_img(os.path.join(scan_dir, cine_file[1]))
temp_sel_cine = np.squeeze(cine_scan)[:, :, ::-1, ::-1]
hplotc.SlidingPlot(temp_sel_cine)
n_coil, n_time, n_phase, n_freq = temp_sel_cine.shape

sense_scan, _ = read_scan.read_cpx_img(os.path.join(scan_dir, sense_ref_file[0]))

summed_sense = np.squeeze(np.abs(sense_scan).sum(axis=0)[0])
hplotc.SlidingPlot(np.moveaxis(summed_sense, -1, 0))

sense_factor = 3
n_y, n_z, n_x = np.squeeze(sense_scan).shape[-3:]

# Newly found settings
ymin = 52-38
ymax = 52 + 2 * 38
xmin = 48-29
xmax = 48 + 2*29
z_sel = 59
print(ymin, ymax, xmin, xmax, z_sel)
temp = np.squeeze(sense_scan[:, :, ymin:ymax, :, :, :, :, xmin:xmax, z_sel])

fig = hplotf.plot_3d_list(np.abs(temp).sum(axis=0)[0], augm='np.abs')
fig.axes[0].hlines(temp.shape[-2]//3, 0, temp.shape[-1]-1, colors='r')
fig.axes[0].hlines(2*temp.shape[-2]//3, 0, temp.shape[-1]-1, colors='r')
fig.axes[0].vlines(temp.shape[-1]//3, 0, temp.shape[-2]-1, colors='r')
fig.axes[0].vlines(2*temp.shape[-1]//3, 0, temp.shape[-2]-1, colors='r')

temp_scaled = sktransf.resize(temp.real, (n_coil, 2, n_phase*sense_factor, n_freq)) + \
              1j * sktransf.resize(temp.imag, (n_coil, 2, n_phase*sense_factor, n_freq))

del temp


result_time = []
for sel_time in range(33):
    # sel_time = 0
    sel_cine = temp_sel_cine[:, sel_time]
    n_phase, n_freq = sel_cine.shape[-2:]
    result = np.zeros(temp_scaled.shape[-2:], dtype=complex)
    for i_x in range(n_freq):
        for i in range(n_phase):
            X = temp_scaled[-8:, 0, i::n_phase, i_x] / temp_scaled[0, 1, i::n_phase, i_x]
            b = sel_cine[-8:, i, i_x]
            res = np.matmul(np.linalg.pinv(X), b)

            result[i::n_phase, i_x] = res

    # fig = hplotf.plot_3d_list(result, vmin=(-10000, 10000), augm='np.real')
    # fig.axes[0].hlines(178, 0, 620-1, colors='r')
    # fig.axes[0].hlines(2*178, 0, 620-1, colors='r')
    # fig.axes[0].vlines(620//3, 0, 534-1, colors='r')
    # fig.axes[0].vlines(620*2//3, 0, 534-1, colors='r')

    result_time.append(result)

hplotc.SlidingPlot(np.array(result_time), vmin=(0, 30000))
import imageio
imageio.mimsave('/home/bugger/test.gif', np.abs(np.array(result_time)))


"""
Read data and store it somewehere..
"""
input_dir = '/media/bugger/MyBook/data/7T_scan/prostate_mri_mrl/2021_01_06/pr_16289'
b1_map_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if 'b1map' in x and x.endswith('cpx')]
t2_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if 't2' in x and x.endswith('cpx')]

cpx_img, b1_header = read_scan.read_cpx_img(b1_map_files[0])

b1_map_array = [np.squeeze(read_scan.read_cpx_img(x)[0]) for x in b1_map_files]
t2_array = [np.squeeze(read_scan.read_cpx_img(x)[0]) for x in t2_files]

hplotc.ListPlot([(b1_map_array[0][:, :, 0] / b1_map_array[0][:, :, 1]).sum(axis=0)], augm='np.real')
hplotc.ListPlot(b1_map_array[0].sum(axis=0)[1] - b1_map_array[0].sum(axis=0)[0], augm='np.real')
hplotc.ListPlot([x.sum(axis=0) for x in t2_array], augm='np.abs')

# input_file = '/media/bugger/MyBook/data/7T_scan/prostate_mri_mrl/2021_01_06/pr_16289/pr_06012021_1647041_12_3_t2wV4.cpx'
input_file = '/media/bugger/MyBook/data/7T_scan/prostate_mri_mrl/2021_01_06/pr_16289/pr_06012021_1653458_15_3_t2wV4.cpx'

target_dir = '/home/bugger/Documents/data/7T/test_for_inhomog/prostate_7T'
file_name, ext = os.path.splitext(os.path.basename(input_file))
target_file = os.path.join(target_dir, file_name)

cpx_img, header = read_scan.read_cpx_img(input_file)
np.save(target_file, np.squeeze(cpx_img))

import helper.plot_fun as hplotf
import helper.plot_class as hplotc
np.squeeze(cpx_img.shape)
hplotc.ListPlot(np.squeeze(cpx_img).sum(axis=0), augm='np.abs')
