"""
We are going to try and do a sin/cos basis expansion to estiamte a proper B1 field signal thingy
"""

import helper.array_transf as harray
import reconstruction.ReadCpx as read_cpx
import os
import helper.plot_class as hplotc
import numpy as np
import skimage.transform as sktransf

input_dir = '/media/bugger/MyBook/data/7T_scan/prostate_mri_mrl/2021_01_06/pr_16289'
input_dir = '/media/bugger/MyBook/data/7T_scan/prostate_mri_mrl/2020_07_15/V9_11496'
b1_shim_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if 'b1shim' in x and x.endswith('cpx')]
t2_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if 't2' in x and x.endswith('cpx')]

# Load b1 shim series
b1_shim_array = [np.squeeze(read_cpx.ReadCpx(x).get_cpx_img()) for x in b1_shim_files]
b1_shim_array = np.array([np.rot90(np.array(x), k=1, axes=(-2, -1)) for x in b1_shim_array])

# Load t2w images
t2_array = [np.squeeze(read_cpx.ReadCpx(x).get_cpx_img()) for x in t2_files]
t2_array = np.array([np.rot90(np.array(x), k=1, axes=(-2, -1)) for x in t2_array])

sel_index = 1

# Plot T2 array inhomog patterns
hplotc.ListPlot([np.abs(t2_array[sel_index]).sum(axis=0), np.abs(b1_shim_array[sel_index]).sum(axis=0).sum(axis=0)], augm='np.abs')
hplotc.ListPlot(b1_shim_array[sel_index].sum(axis=0), augm='np.abs', vmin=(0, 100000), ax_off=True, title='transmit')
hplotc.ListPlot(b1_shim_array[sel_index].sum(axis=1), augm='np.abs', vmin=(0, 100000), ax_off=True, title='receive')

# Now we want to find a basis expansion on the B1 shim series and see how CLOSE we can get to the T2 weighted image..
transmit_array = b1_shim_array[sel_index].sum(axis=0)
transmit_array_resize = harray.resize_complex_array(transmit_array, new_shape=(8, 256, 256), preserve_range=True)

target_array = np.abs(t2_array[sel_index]).sum(axis=0)
target_array_resize = sktransf.resize(target_array, (256, 256), preserve_range=True)
hplotc.ListPlot([target_array_resize, transmit_array_resize.sum(axis=0)], augm='np.abs')

target_mask = harray.get_treshold_label_mask(target_array_resize, treshold_value=0.4 * np.mean(target_array_resize))
temp = np.abs(transmit_array_resize).sum(axis=0)
transmit_mask = harray.get_treshold_label_mask(temp, treshold_value=0.2 * np.mean(temp))
hplotc.ListPlot([target_mask, transmit_mask], augm='np.abs')

target_affine_coords, target_crop_coords = harray.get_center_transformation_coords(target_mask)
target_array_recenter = harray.apply_center_transformation(target_array_resize,
                                                           affine_coords=target_affine_coords,
                                                           crop_coords=target_crop_coords)

transmit_affine_coords, transmit_crop_coords = harray.get_center_transformation_coords(transmit_mask)
transmit_array_recenter = np.array([harray.apply_center_transformation(x,
                                                                       affine_coords=target_affine_coords,
                                                                       crop_coords=target_crop_coords,
                                                                       dtype=np.complex64) for x in transmit_array_resize])

hplotc.ListPlot([target_array_recenter, transmit_array_recenter.sum(axis=0)], augm='np.abs')

# Shim the transmit thing...
import tooling.shimming.b1shimming_single as mb1
b1_mask = np.zeros(transmit_array_recenter.shape[-2:])
n_y, n_x = b1_mask.shape
y_center, x_center = (n_y // 2, n_x // 2)
delta_x = int(0.10 * n_y)
b1_mask[y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x] = 1
shim_proc = mb1.ShimmingProcedure(transmit_array_recenter, b1_mask)
opt_shim, opt_value = shim_proc.find_optimum()
transmit_array_shim = harray.apply_shim(transmit_array_recenter, opt_shim)

hplotc.ListPlot([transmit_array_recenter.sum(axis=0), transmit_array_shim], augm='np.abs')

global_array = np.abs(transmit_array_shim) / (np.abs(np.mean(transmit_array_shim * b1_mask)))
global_array = harray.scale_minmax(global_array) * np.pi/2
target_array = harray.scale_minmax(target_array_recenter)

# Inspect the 'shimmed' array
hplotc.ListPlot([target_array, global_array], augm='np.abs')

def calc_stuff(x):
    global global_array
    global target_array
    res = calc_fun(x, global_array)
    diff = np.mean((res - target_array_recenter) ** 2)
    return diff

def calc_fun(x, inp_array):
    # list_of_fun = np.array([inp_array, np.sin(inp_array), np.cos(inp_array),
    #                np.sin(inp_array) ** 2, np.cos(inp_array) ** 2,
    #                np.sin(inp_array) ** 3, np.cos(inp_array) ** 3,
    #                np.sin(inp_array) * np.cos(inp_array), np.sin(inp_array) **2 * np.cos(inp_array) ** 2])
    # list_of_fun = np.array([inp_array, np.sin(inp_array), np.sin(inp_array) ** 3, np.sin(inp_array) ** 5])
    list_of_fun = np.array([np.sin(inp_array), np.sin(inp_array) ** 3])
    res = np.einsum("cxy, c -> xy", list_of_fun, x) / len(list_of_fun)
    return res

x_initial = np.random.rand(2)
import scipy.optimize
amp_bound = [(0, 1) for _ in range(len(x_initial))]
opt_obj = scipy.optimize.minimize(fun=calc_stuff, x0=x_initial, tol=1e-8,
                                        method='Powell', bounds=amp_bound,
                                        options={"maxiter": 1000,
                                                 "disp": True})
print(opt_obj.x)
res = calc_fun(opt_obj.x, global_array)
hplotc.ListPlot([res, global_array, target_array])
temp = b1_shim_array[0].sum(axis=0)



import importlib
import scipy.io
import h5py
ddata = '/home/bugger/Documents/data/test_clinic_registration/flavio_data/M01.mat'
temp = np.moveaxis(scipy.io.loadmat(ddata)['Model']['B1plus'][0][0], -1, 0)
temp = harray.scale_minmax(temp, is_complex=True)

ddata = '/home/bugger/Documents/data/check_clinic_registration/M23_to_46_MR_20200925_0002.h5'
temp = np.array(h5py.File(ddata, 'r')['data'][15])
temp = harray.scale_minmax(temp, is_complex=True)
hplotc.SlidingPlot(temp)

importlib.reload(shim_tool)
b1_obj = shim_tool.B1ManualShimming(temp, initial_angle=np.pi)

import tooling.shimming.b1shimming_single as mb1
b1_mask = np.zeros(temp.shape[-2:])
n_y, n_x = b1_mask.shape
y_center, x_center = (n_y // 2, n_x // 2)
delta_x = int(0.3 * n_y)
b1_mask[y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x] = 1
shim_proc = mb1.ShimmingProcedure(temp, b1_mask)
opt_shim, opt_value = shim_proc.find_optimum()
temp_shim = harray.apply_shim(temp, opt_shim)
mean_value = temp_shim[y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x].mean()
temp_sin = np.sin(np.abs(temp_shim) * np.pi/2) ** 3
# temp_sin = harray.scale_minmax(temp_sin)
hplotc.ListPlot([temp_shim * (1+b1_mask), temp_sin], augm='np.abs')

