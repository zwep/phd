import numpy as np
import helper.array_transf as harray
import sigpy.mri
import matplotlib.pyplot as plt
import scipy.io
import helper.misc as hmisc
import helper.plot_class as hplotc
import os
import helper.reconstruction as hrecon
import reconstruction.ReadCpx as read_cpx

"""
Yes, here we can see that nufft stuff and the desired spokes worked!
"""

base_dir = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_12_01/bart_data'
scan_base_dir = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_12_01/ca_29045'

dsorted = os.path.join(base_dir, 'bart_17_2_sorted.mat')
sorted_array = scipy.io.loadmat(dsorted)['bart_17_2_sorted']
sel_sin_file = os.path.join(scan_base_dir, 'ca_01122021_1019026_17_2_transverse_retro_radialV4.sin')
trajectory = hrecon.get_trajectory_sin_file(sel_sin_file)

# hrecon.get_key_from_sin_file()

dcf = np.sqrt(trajectory[:, :, 0] ** 2 + trajectory[:, :, 1] ** 2)
n_coil = sorted_array.shape[3]
n_card = sorted_array.shape[5]
result = []
for i_card in range(n_card):
    temp = []
    for i_coil in range(n_coil):
        sel_array = sorted_array[:, :, 0, i_coil, 0, i_card]
        # First number of spokes, then numbers of points,,
        sel_array = np.moveaxis(sel_array, -1, 0)
        temp_img = sigpy.nufft_adjoint(sel_array * dcf, coord=trajectory)
        temp.append(temp_img)
    result.append(temp)

A = np.array(result)
hplotc.SlidingPlot(np.abs(A).sum(axis=1))

# Now try with fantom...
dunsorted = '/media/bugger/WORK_USB/fantom_data.mat'
sel_sin_file = "/media/bugger/WORK_USB/se_26726/se_20102021_1910181_25_2_surveylr_10_phasesV4.sin"
unsorted_array = scipy.io.loadmat(dunsorted)['unsorted_data']
trajectory = hrecon.get_trajectory_sin_file(sel_sin_file)
ovs = float(hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_grid_overs_factor'))
width = int(hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_grid_kernel_size'))
n_coil = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_channel_names'))
n_card = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_cardiac_phases'))


result = []
for i_coil in range(n_coil):
    selected_data = unsorted_array[:, i_coil::n_coil][:, :trajectory.shape[0]]
    selected_data = np.moveaxis(selected_data, -1, 0)
    temp_img = sigpy.nufft_adjoint(selected_data, coord=trajectory)
    result.append(temp_img)

result = np.array(result)
hplotc.ListPlot(np.abs(result).sum(axis=0, keepdims=True), augm='np.abs')


"""
Now try it with unsorted Bart data
"""

dunsorted = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_12_01/bart_data/bart_17_2_unsorted.mat'
unsorted_array = scipy.io.loadmat(dunsorted)['bart_17_2_unsorted']
sel_sin_file = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_12_01/ca_29045/ca_01122021_1019026_17_2_transverse_retro_radialV4.sin'
trajectory = hrecon.get_trajectory_sin_file(sel_sin_file)
ovs = float(hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_grid_overs_factor'))
width = int(hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_grid_kernel_size'))
n_coil = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_channel_names'))
n_card = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_cardiac_phases'))
n_spokes = trajectory.shape[0]
result = []
for i_coil in range(n_coil):
    temp = []
    for i_card in range(n_card):
        # okay WHAT am I doing here.
        # First of all: We have unsorted data. This means that the spokes are in chronological order
        # Hence we have ``blocks`` of n_coil lines. Therefore we do i_coil::n_coil.
        # Then n_spokes for each cardiac phase. Assuming we have acquired all the spokes nicely and every cardiac phase comes in order etc.
        # Then we can do i_spoke::n_card.
        # To make sure that we can use the data.. we limit it to n_spokes. Then the trajectory variable makes sense.
        selected_data = unsorted_array[:, i_coil::n_coil][:, i_card::n_card][:, :n_spokes]
        selected_data = np.moveaxis(selected_data, -1, 0)
        temp_img = sigpy.nufft_adjoint(selected_data, coord=trajectory, oversamp=ovs, width=width)
        temp.append(temp_img)
    result.append(temp)

result = np.array(result)

hplotc.ListPlot(np.abs(result).sum(axis=0)[0], augm='np.abs')



result_single_phase = []
for i_coil in range(n_coil):
    selected_data = unsorted_array[:, i_coil::n_coil][:, :n_spokes]
    selected_data = np.moveaxis(selected_data, -1, 0)
    temp_img = sigpy.nufft_adjoint(selected_data, coord=trajectory, oversamp=ovs, width=width)
    result_single_phase.append(temp_img)

result_single_phase = np.array(result_single_phase)
hplotc.ListPlot(np.abs(result_single_phase).sum(axis=0, keepdims=True), augm='np.abs')