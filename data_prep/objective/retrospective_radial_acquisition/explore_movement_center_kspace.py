
import os
import scipy.io
import numpy as np
import helper.plot_class as hplotc
import matplotlib.pyplot as plt


"""
Got some unsorted data

Lets check it out

What do I have...

Data from Fantom (nov)
Data from Matthijs
Data from Bart
Data from Fantom (dec)

1. I can determine the angle of the spokes that has been used
Although I was note sure when doing the cardiac thing..

2. I dont know why the number of spokes is so low
    Check sorted data from Fantom (nov)

3. I need to know how to read the Mark in the Physlog and how to determine which part
corresponds to the scanner

3a. Then I can compare the signal in the center with that physlog.

==== solutions ====

1.

2.

3. Partially solved. I have found ways to use the marker and can check it against the respitory data.
I dont have enough time resolution for the cardiac movement

"""

"""
What I am doing here is checking what kind of movement I can spot in the
untriggered cardiac cases...
"""


center_signal = np.mean(np.abs(untrig_radial_high_time[536//2-10:536//2+10, ::24]), axis=0)
fig, ax = plt.subplots()
ax.plot(center_signal)

result_fft = np.abs(np.fft.fft(center_signal))
result_fft_freq = np.fft.fftfreq(len(center_signal), d=4*10**-3)
fig, ax = plt.subplots()
# I see A peak near 1... but is that OK..??
ax.plot(result_fft_freq, result_fft)

import reconstruction.ReadPhyslog as read_phys
phys_log_high_time = read_phys.ReadPhyslog('/media/bugger/MyBook/data/7T_scan/cardiac/2021_05_02/V9_19531/SCANPHYSLOG_v9_02052021_1432554_7_1_transradialfast_high_timeV4.log')
ppu_signal = phys_log_high_time.phys_log_dict['ppu']

# found a sample rate of 2 ms... Looks OK
plt.plot(np.arange(0, len(center_signal)) * 2 * 10 ** -3, ppu_signal)

# Here I am checking if I can see any relevant frequencies
result_fft = np.abs(np.fft.fft(center_signal))
result_fft_freq = np.fft.fftfreq(len(center_signal), d=4*10**-3)
fig, ax = plt.subplots()
# I see A peak near 1... but is that OK..??
ax.plot(result_fft_freq, result_fft)



# Check the spokes from different acquisitions...
# They are all the same
import scipy.io
ddata_spokes_trans = '/media/bugger/MyBook/data/7T_data/cardiac_radial/V9_13975/v9_04112020_1726334_7_2_transradialfastV4_spokes.mat'
ddata_spokes_2ch = '/media/bugger/MyBook/data/7T_data/cardiac_radial/V9_13975/v9_04112020_1742232_10_2_p2chradialV4_spokes.mat'
ddata_spoke_4ch = '/media/bugger/MyBook/data/7T_data/cardiac_radial/V9_13975/v9_04112020_1759228_14_2_4chradialV4_spokes.mat'

spokes_obj_trans = scipy.io.loadmat(ddata_spokes_trans)['kpos_data']
spokes_obj_2ch = scipy.io.loadmat(ddata_spokes_2ch)['kpos_data']
spokes_obj_4ch = scipy.io.loadmat(ddata_spoke_4ch)['kpos_data']
plt.plot(spokes_obj_2ch[:, 1, 0, 0], 'b')
plt.plot(spokes_obj_2ch[:, 1, 0, 1], '--b')
plt.plot(spokes_obj_trans[:, 1, 0, 0], 'r')
plt.plot(spokes_obj_trans[:, 1, 0, 1], '--r')
plt.plot(spokes_obj_4ch[:, 1, 0, 0], 'k')
plt.plot(spokes_obj_4ch[:, 1, 0, 1], '--k')

# Check spokes of different undersampled acquisitions
ddata_spoke_trans_dyn = '/media/bugger/MyBook/data/7T_data/cardiac_radial/V9_16051/v9_09122020_1705519_8_2_transradialfastdynV4_spokes.mat'
ddata_spoke_trans = '/media/bugger/MyBook/data/7T_data/cardiac_radial/V9_16051/v9_09122020_1704205_7_2_transradialfastV4_spokes.mat'
# The same spokes...
spokes_obj_trans_dyn = scipy.io.loadmat(ddata_spoke_trans_dyn)['kpos_data']
spokes_obj_trans = scipy.io.loadmat(ddata_spoke_trans)['kpos_data']
plt.plot(spokes_obj_trans_dyn[:, 1, 0, 0], 'b', alpha=0.5)
plt.plot(spokes_obj_trans_dyn[:, 1, 0, 1], '--b', alpha=0.5)
plt.plot(spokes_obj_trans[:, 1, 0, 0], 'r', alpha=0.5)
plt.plot(spokes_obj_trans[:, 1, 0, 1], '--r', alpha=0.5)
for i in range(0, 401, 10):
    plt.scatter(spokes_obj_trans[:, i, 0, 0], spokes_obj_trans[:, i, 0, 1])

for i in range(0, 401, 1):
    complex_vector = spokes_obj_trans[:, i, 0, 0] + 1j * spokes_obj_trans[:, i, 0, 1]
    plt.scatter(i, min(np.angle(complex_vector)), color='k')
    plt.scatter(i, max(np.angle(complex_vector)), color='r')
    print(min(np.angle(complex_vector)), max(np.angle(complex_vector)))

# Check out the difference in matrix size of the acquisition.. and the spokes...
ddata_spoke_trans = '/media/bugger/MyBook/data/7T_data/cardiac_radial/V9_16051/v9_09122020_1704205_7_2_transradialfastV4_spokes.mat'
ddata_trans = '/media/bugger/MyBook/data/7T_data/cardiac_radial/V9_16051/v9_09122020_1704205_7_2_transradialfastV4.mat'
spokes_obj_trans = scipy.io.loadmat(ddata_spoke_trans)['kpos_data']
obj_trans = scipy.io.loadmat(ddata_trans)['reconstructed_data']
print('Radial spoke shape ', spokes_obj_trans.shape)
print('Radial object shape ', obj_trans.shape)


# Lets check if we can recollect the dynamic image... Just to be sure about our spoke order...
# First guess is this
first_dyn_first_coil_kspace = unsorted_array[:, ::24][:, :66]
hplotc.ListPlot(first_dyn_first_coil_kspace, augm='np.abs')

first_dyn_first_coil_kspace = unsorted_array[:, ::24][:, 66:2*66]
hplotc.ListPlot(first_dyn_first_coil_kspace, augm='np.abs')

# I dont know how stuff gets undersampled.. but I need 66 spokes out of these things...
import sigpy
import sigpy.mri
img_shape = (536, 536)
ovs = 1.25
width = 6
n_points = max(img_shape)
n_spokes = 66
# Define trajectory..
trajectory_radial = sigpy.mri.radial(coord_shape=(n_spokes, n_points, 2), img_shape=img_shape, golden=False)
import matplotlib
cmap = matplotlib.cm.get_cmap('Reds')
fig, ax = plt.subplots(2)
for ii, itraj in enumerate(trajectory_radial):
    temp = itraj[:, 0] + 1j * itraj[:, 1]
    ax[0].scatter(itraj[:, 0], itraj[:, 1], c=np.array(cmap(ii / len(trajectory_radial))).reshape(1, -1))
    ax[1].scatter(ii, min(np.angle(temp)), color='k')
    ax[1].scatter(ii, max(np.angle(temp)), color='r')
# n spokes, n points, x/y
# Range goes from -npoints/2, npoints/2

trajectory_radial = trajectory_radial.reshape(-1, 2)

# We might remove this one..? Because it is so generic and repetitive over all the spokes
dcf = np.sqrt(trajectory_radial[:, 0] ** 2 + trajectory_radial[:, 1] ** 2)
# print('Device on ksacpe' ,sigpy.get_device(temp_kspace), type(temp_kspace))
# print('Coords', sigpy.get_device(undersampled_trajectory), type(undersampled_trajectory))
temp_reshape = (first_dyn_first_coil_kspace.T).reshape(-1)
temp_img = sigpy.nufft_adjoint((first_dyn_first_coil_kspace.T).reshape(-1) * dcf, coord=trajectory_radial, oshape=img_shape, width=width, oversamp=ovs)
hplotc.ListPlot(temp_img, augm='np.abs')
#
# Load the CPX file of `sel_file` content to check how many cardiac phases we had
import reconstruction.ReadCpx as read_cpx
dir_cpx = '/media/bugger/MyBook/data/7T_data/cardiac_unsorted_data/V9_19531/scanned_files/v9_02052021_1431531_6_3_transradialfastV4.cpx'
cpx_obj = read_cpx.ReadCpx(dir_cpx)
img_array = cpx_obj.get_cpx_img()
n_coils = 24
n_cardiac = 8
n_spoke = signal_middle_mean.shape[0] / n_coils / n_cardiac

# Load the physlog
import reconstruction.ReadPhyslog as read_phys
dir_phys = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_05_02/V9_19531/SCANPHYSLOG_v9_02052021_1431531_6_1_transradialfastV4.log'
phys_obj = read_phys.ReadPhyslog(dir_phys)
index_ppu = phys_obj.columns.index('ppu')
plt.plot(phys_obj.phys_log[:, index_ppu])

# Find the peaks in the physlog
import scipy.signal
res = scipy.signal.find_peaks(phys_obj.phys_log[:, index_ppu])
peak_location = res[0]
peak_height = phys_obj.phys_log[peak_location, index_ppu]
index_peak_height_filtered = peak_height > 1000
filtered_peak = peak_location[index_peak_height_filtered]
filtered_height = peak_height[index_peak_height_filtered]

# Plot the phys log with peak detection...
plt.plot(phys_obj.phys_log[:, index_ppu])
plt.scatter(filtered_peak, filtered_height, c='r')
n_peaks = len(filtered_peak)

# Perform moving average and check frequency of physlog
import helper.array_transf as harray
moving_average = harray.moving_average(phys_obj.phys_log[:, index_ppu], 1000)
plt.figure()
plt.plot(moving_average)

physlog_fourier = np.fft.fft(phys_obj.phys_log[:, index_ppu])
physlog_fourier_ma = np.fft.fft(moving_average)
plt.plot(np.abs(physlog_fourier), 'k')
plt.plot(np.abs(physlog_fourier_ma), 'r')

"""
Now that we know how this could work... lets try it again.
"""

import numpy as np
import helper.array_transf as harray
import sigpy.mri
import matplotlib.pyplot as plt
import scipy.io
import helper.misc as hmisc
import helper.plot_class as hplotc
import os
import matplotlib.pyplot as plt
import helper.reconstruction as hrecon


dunsorted = '/media/bugger/WORK_USB/bart_data/bart_17_2_unsorted.mat'
unsorted_array = scipy.io.loadmat(dunsorted)['bart_17_2_unsorted']
sel_sin_file = '/media/bugger/WORK_USB/2021_12_01/ca_29045/ca_01122021_1019026_17_2_transverse_retro_radialV4.sin'
ovs = float(hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_grid_overs_factor'))
width = int(hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_grid_kernel_size'))
n_coil = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_channel_names'))
n_card = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_cardiac_phases'))
n_spokes = int(hrecon.get_spokes_from_sin_file(sel_sin_file))

x_img = unsorted_array.shape[0]

kspace_data_17 = []
for i_coil in range(n_coil):
    selected_data = unsorted_array[:, i_coil::n_coil][:, :n_spokes*n_card]
    kspace_data_17.append(selected_data)

kspace_data_17 = np.array(kspace_data_17)

center_signal_17 = kspace_data_17[:, x_img//2]
plt.plot(np.abs(center_signal_17).T)

kspace_data_17_fft = np.array([np.abs(np.fft.fft(x)) for x in kspace_data_17[:, x_img//2]])
kspace_data_17_fft_freq = np.fft.fftfreq(len(kspace_data_17[0, 0, :]), d=0.0045)

for ii, ifft in enumerate(kspace_data_17_fft):
    fig, ax = plt.subplots()
    ax.plot(kspace_data_17_fft_freq, ifft)
    freq_first = kspace_data_17_fft_freq[1]
    ifft_first = ifft[1]
    ifft_max = ifft.max()
    ax.plot(freq_first, ifft_first, 'ro')
    print(ii, round(freq_first, 2), round(ifft_first), round(ifft_first / ifft.mean(), 2), round(ifft_max))


import scipy.signal
res = scipy.signal.find_peaks(result_fft[0])
peak_location = res[0]
peak_abs = result_fft[0][peak_location]

peak_tresh_index = np.argwhere(peak_abs > 2700)
peak_freq_filt = result_fft_freq[peak_location][peak_tresh_index]
peak_abs_filt = result_fft[0][peak_location][peak_tresh_index]

plt.plot(result_fft_freq, result_fft[0], 'b--')
plt.plot(peak_freq_filt, peak_abs_filt, 'r*')
np.diff(peak_freq_filt.T)

# Cycles / second
# Okay... dummy thing..
# Yes now I understand it...
T = 10
dt = 0.001
t_range = np.arange(0, T, dt)
freq0 = 4
freq1 = 2
y = np.sin(2*np.pi * freq0 * t_range) + np.sin(2*np.pi * freq1 * t_range)
fig, ax = plt.subplots(2)
ax[0].plot(t_range, y)

# Now check the Fourier transform.... Where does it peak?
result_fft = np.abs(np.fft.fft(y))
result_fft_freq = np.fft.fftfreq(len(t_range), d=dt)
ax[1].plot(result_fft_freq, result_fft)
ax[1].set_xlim(-5, 5)

recovered = np.fft.ifft(np.fft.fft(y))
fig, ax = plt.subplots(2)
ax[0].plot(np.abs(recovered))


"""
We got some physlog from Bart as well.. 
Lets plot that and the information from the spokes..
"""

import matplotlib.pyplot as plt
import numpy as np
import reconstruction.ReadPhyslog as read_phys
import importlib
importlib.reload(read_phys)

ddata_physlog_bart_14 = '/media/bugger/WORK_USB/2021_12_01/ca_29045/SCANPHYSLOG_ca_01122021_1016141_14_2_transverse_retro_radialV4.log'
ddata_physlog_bart_15 = '/media/bugger/WORK_USB/2021_12_01/ca_29045/SCANPHYSLOG_ca_01122021_1016537_15_2_transverse_retro_radialV4.log'
ddata_physlog_bart_16 = '/media/bugger/WORK_USB/2021_12_01/ca_29045/SCANPHYSLOG_ca_01122021_1017533_16_2_transverse_retro_radialV4.log'
ddata_physlog_bart_17 = '/media/bugger/WORK_USB/2021_12_01/ca_29045/SCANPHYSLOG_ca_01122021_1019026_17_2_transverse_retro_radialV4.log'
ddata_physlog_bart_18 = '/media/bugger/WORK_USB/2021_12_01/ca_29045/SCANPHYSLOG_ca_01122021_1020068_18_1_transverse_dyn_20p_radial_no_triggerV4.log'
ddata_physlog_bart_sense = '/media/bugger/WORK_USB/2021_12_01/ca_29045/SCANPHYSLOG_ca_01122021_0940162_4_1_cine1slicer2_traV4.log'

physlog_list = [ddata_physlog_bart_14, ddata_physlog_bart_15, ddata_physlog_bart_16, ddata_physlog_bart_17, ddata_physlog_bart_18, ddata_physlog_bart_sense]
fig, ax = plt.subplots(len(physlog_list))
for ii, iphys in enumerate(physlog_list):
    phys_log_high_time = read_phys.ReadPhyslog(iphys)
    physlog_ppu = phys_log_high_time.phys_log_dict['ppu']
    sample_rate = 2 * 10 ** -3
    t_range = np.arange(0, len(physlog_ppu)) * sample_rate
    ax[ii].plot(t_range, physlog_ppu)
    ax[ii].set_ylim(-6000, 6000)
    # ax[ii].set_xlim(90, 100)


"""
...
"""
import os
import matplotlib.pyplot as plt
import scipy.io
import helper.reconstruction as hrecon
import numpy as np
import reconstruction.ReadPhyslog as read_phys

base_dir = '/home/bugger/Documents/data/7T/cardiac/bart_data'
dunsorted = os.path.join(base_dir, 'bart_17_2_unsorted.mat')
sel_sin_file = os.path.join(base_dir, 'ca_01122021_1019026_17_2_transverse_retro_radialV4.sin')
sel_phys_log = os.path.join(base_dir, 'SCANPHYSLOG_ca_01122021_1019026_17_2_transverse_retro_radialV4.log')
unsorted_array = scipy.io.loadmat(dunsorted)['bart_17_2_unsorted']

n_points = unsorted_array.shape[0]
delta_center = int(0.05 * n_points)
ovs = float(hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_grid_overs_factor'))
width = int(hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_grid_kernel_size'))
n_coil = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_channel_names'))
n_card = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_cardiac_phases'))
# n_spokes = int(hrecon.get_spokes_from_sin_file(sel_sin_file))
n_spokes = 260
TR = 4.5 * 10 ** -3
print('ncoil, ncard, nspokes')
print(n_coil, n_card, n_spokes)

x_img = unsorted_array.shape[0]

kspace_data_17 = []
for i_coil in range(n_coil):
    selected_data = unsorted_array[:, i_coil::n_coil]# [:, :n_spokes*n_card]
    kspace_data_17.append(selected_data)

kspace_data_17 = np.array(kspace_data_17)
center_signal_17 = np.mean(np.abs(kspace_data_17[:, x_img//2 -delta_center:x_img//2 +delta_center]), axis=1)
plt.plot(np.arange(0, n_spokes)* 4.46 * 10 ** -3, np.abs(center_signal_17).T)

# Hier zie je variatie in.. ook omdat je steeds een andere hoek hebt per spoke....
fig, ax = plt.subplots(len(center_signal_17[-8:]))
for ii, i_coil in enumerate(center_signal_17[-8:]):
    ax[ii].plot(np.arange(0, n_spokes) * 4.46 * 10 ** -3, np.abs(i_coil))


# Now read the PHYSLOG

phys_obj = read_phys.ReadPhyslog(sel_phys_log)
import collections
ppu_data = phys_obj.phys_log_dict['ppu']
plot_max = 7000
fig, ax = plt.subplots(1)
ax.plot(ppu_data)
ax.set_ylim(-plot_max, plot_max)
mark_counter = collections.Counter(phys_obj.phys_log_dict['mark'])
# These are too common.
# 0000 - normal data entry
# 0002 - cardiac trigger
ignored_keys = ["0000", "0002"]
for i_key, i_mark in mark_counter.items():
    if i_key not in ignored_keys:
        mark_index = np.argwhere([x == i_key for x in phys_obj.phys_log_dict['mark']])
        mark_index = mark_index.ravel()
        for i_mark_index in mark_index:
            ax.vlines(x=i_mark_index, ymin=-plot_max, ymax=plot_max, color='k', alpha=0.5)
            ax.text(x=i_mark_index, y=int(0.9 * plot_max), s=i_key)

# How to get the last bit...
acquisition_mark = "0008"
sample_rate = 2 * 10 ** -3
mark_array = np.array(phys_obj.phys_log_dict['mark'])
ppu_array = np.array(phys_obj.phys_log_dict['ppu'])
index_acquisition_mark = np.argwhere(mark_array == acquisition_mark).ravel()
difference_index = np.diff(index_acquisition_mark)
highest_difference = np.argwhere(difference_index > np.mean(difference_index)).ravel()[0]

plt.plot(index_acquisition_mark)
plt.plot(highest_difference, index_acquisition_mark[highest_difference], 'ro')
# This is how to get the last PPU stuff indices then..
last_active_index_array = index_acquisition_mark[highest_difference+1:]
starting_index = last_active_index_array[0]
final_index = last_active_index_array[-1]
t_range = np.arange(0, final_index - starting_index) * sample_rate
# Plot the last acquired PPYU stuff. Porlly belongs to the acuiqred data
plt.figure()
plt.plot(t_range, ppu_array[starting_index:final_index])


"""
This is going to be one messy file........

I scanned a fantom with varying settings.. lets see how many spokes we get...
And is it the same amount as we expect
"""

import re
import scipy.io
import helper.plot_class as hplotc
import numpy as np
import h5py
import os
import helper.reconstruction as hrecon

ddata = '/media/bugger/WORK_USB/2021_12_09/mat_data'
list_files = [os.path.join(ddata, x) for x in os.listdir(ddata)]
unsorted_list = []
for x in list_files:
    A = scipy.io.loadmat(x)['unsorted_data']
    unsorted_list.append(A)
    file_name, _ = os.path.splitext(x)
    print(file_name)
    sin_file = file_name + '.sin'
    sin_file = re.sub('mat_data', 'ca_29447', sin_file)
    trajectory = hrecon.get_trajectory_sin_file(sin_file)
    n_spokes = trajectory.shape[0]
    ovs = float(hrecon.get_key_from_sin_file(sin_file, 'non_cart_grid_overs_factor'))
    width = int(hrecon.get_key_from_sin_file(sin_file, 'non_cart_grid_kernel_size'))
    n_coil = int(hrecon.get_key_from_sin_file(sin_file, 'nr_channel_names'))
    n_card = int(hrecon.get_key_from_sin_file(sin_file, 'nr_cardiac_phases'))
    print(n_spokes, n_coil, n_card)

"""
Okay now get the 
"""

ddata_not_sorted = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_11_24/recon_data.mat'
non_sorted_array = scipy.io.loadmat(ddata_not_sorted)['recon_data']
non_sorted_data_array = non_sorted_array[0][0]
non_sorted_data_array.shape


"""

Check Barts unsorted 100% radial data
And check his Physlog stuff
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import reconstruction.ReadPhyslog as read_phys
import scipy.io
import helper.reconstruction as hrecon
import helper.plot_fun as hplotf
import reconstruction.ReadCpx as read_cpx

# Define paths
# dbart = '/home/bugger/Documents/data/7T/cardiac/bart_data'
dbart = '/home/bugger/Documents/data/7T/cardiac/matthuijs_data/2022_01_19'
# unsorted_100p_data = 'untriggered_dyn_100_unsorted_data.mat'
# unsorted_100p_phys = 'SCANPHYSLOG_ca_01122021_1021250_19_1_transverse_dyn_100p_radial_no_triggerV4.log'
# unsorted_100p_sin = 'ca_01122021_1021250_19_2_transverse_dyn_100p_radial_no_triggerV4.sin'
# sorted_100p_cpx = 'ca_01122021_1021250_19_3_transverse_dyn_100p_radial_no_triggerV4.cpx'

dbart = '/home/bugger/Documents/data/7T/cardiac/matthuijs_data/2022_01_19/derp_data'
unsorted_100p_data = 'ca_19012022_1658096_8_2_transradial_no_trigger_100V4.mat'
unsorted_100p_phys = 'SCANPHYSLOG_ca_19012022_1652538_6_1_transradialdyn_100V4.log'
unsorted_100p_sin = 'ca_19012022_1658096_8_2_transradial_no_trigger_100V4.sin'
sorted_100p_cpx = 'ca_19012022_1658096_8_3_transradial_no_trigger_100V4.cpx'

# Read the data
phys_obj_100p = read_phys.ReadPhyslog(os.path.join(dbart, unsorted_100p_phys))
mat_obj = scipy.io.loadmat(os.path.join(dbart, unsorted_100p_data))['unsorted_data']

# Get hyper parameters
n_coil = int(hrecon.get_key_from_sin_file(os.path.join(dbart, unsorted_100p_sin), 'nr_channel_names'))
n_radial_points, n_lines = mat_obj.shape
TR = float(hrecon.get_key_from_sin_file(os.path.join(dbart, unsorted_100p_sin), 'repetition_times')) * 10 ** -3
n_lines_per_coil = n_lines // n_coil

# Split data into useable chunks
coil_lines = np.array([mat_obj[:, i::n_coil] for i in range(n_coil)])
center_coil_lines = coil_lines[:, n_radial_points//2]

time_range_radial = np.arange(0, n_lines_per_coil) * TR
# Plot the acquired data
hplotf.plot_multi_lines(np.abs(center_coil_lines.T), x_range=time_range_radial)
fig, ax = plt.subplots(8)
for ii in range(8, 0, -1):
    ax[ii-1].plot(time_range_radial, np.imag(center_coil_lines[-ii]))

# Plot the physlog data
import collections
ppu_data = phys_obj_100p.phys_log_dict['ppu']
plot_max = 7000
frequency_physlog = 2 * 10 ** -3  # 2ms
n_physlog_points = len(ppu_data)
physlog_time = np.arange(0, n_physlog_points) * frequency_physlog

# Plot the physlog data
fig, ax = plt.subplots(1)
ax.plot(physlog_time, ppu_data, c='r')
ax.set_ylim(-plot_max, plot_max)
mark_counter = collections.Counter(phys_obj_100p.phys_log_dict['mark'])
# These are too common.
# 0000 - normal data entry
# 0002 - cardiac trigger
ignored_keys = ["0000", "0002"]
for i_key, i_mark in mark_counter.items():
    if i_key not in ignored_keys:
        mark_index = np.argwhere([x == i_key for x in phys_obj_100p.phys_log_dict['mark']])
        mark_index = mark_index.ravel()
        for i_mark_index in mark_index:
            ax.vlines(x=physlog_time[i_mark_index], ymin=-plot_max, ymax=plot_max, color='k', alpha=0.5)
            ax.text(x=physlog_time[i_mark_index], y=int(0.9 * plot_max), s=i_key)

# Now check the actual data
cpx_obj = read_cpx.ReadCpx(os.path.join(dbart, sorted_100p_cpx))
untriggered_100p_array = cpx_obj.get_cpx_img()
hplotc.SlidingPlot(np.abs(untriggered_100p_array).sum(axis=0))