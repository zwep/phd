"""
We did a special acquisition with only a single spoke for a long time...
Lets check what the data can tell us...
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import helper.reconstruction as hrecon
import helper.array_transf as harray
import scipy.io
import matplotlib.pyplot as plt
import reconstruction.ReadPhyslog as read_phys

ddata = '/media/bugger/MyBook/data/7T_scan/cardiac/2022_01_19/derp_data/ca_19012022_1654536_7_2_transradialdyn_1V4.mat'
dphys = '/media/bugger/MyBook/data/7T_scan/cardiac/2022_01_19/derp_data/SCANPHYSLOG_ca_19012022_1654536_7_1_transradialdyn_1V4.log'
dsin = '/media/bugger/MyBook/data/7T_scan/cardiac/2022_01_19/ca_30409/ca_19012022_1654536_7_2_transradialdyn_1V4.sin'
single_acq_line = scipy.io.loadmat(ddata)['unsorted_data']
physlog_obj = read_phys.ReadPhyslog(dphys)

# Get scan parameters
n_coil = int(hrecon.get_key_from_sin_file(os.path.join(ddata, dsin), 'nr_channel_names'))
n_radial_points, n_lines = single_acq_line.shape
TR = float(hrecon.get_key_from_sin_file(os.path.join(ddata, dsin), 'repetition_times')) * 10 ** -3
n_lines_per_coil = n_lines // n_coil
scan_time = np.arange(0, n_lines_per_coil) * TR
coil_lines = np.array([single_acq_line[:, i::n_coil] for i in range(n_coil)])
center_index = n_radial_points // 2
time_coil_data = np.arange(n_lines_per_coil) * TR

# Get the ppu data
physlog_obj.visualize_label('ppu')

ppu_data = physlog_obj.phys_log_dict['ppu']
sample_rate_physlog = physlog_obj.sample_rate
stop_mark_index = np.argwhere([x == physlog_obj.stop_mark for x in physlog_obj.phys_log_dict['mark']])[0][0]
start_scan_index = stop_mark_index - int(n_lines_per_coil * TR / physlog_obj.sample_rate)
scan_ppu_data = np.array(ppu_data[start_scan_index:stop_mark_index])
time_ppu_data = np.arange(stop_mark_index - start_scan_index) * sample_rate_physlog

# Visualize the single spoke data with the PPU
sub_sel_coil = np.arange(-8, 0)
fig, ax = plt.subplots(len(sub_sel_coil))
for i_coil in sub_sel_coil:
    ax[abs(i_coil)-1].twinx().plot(time_coil_data, coil_lines[i_coil, center_index], 'b')
    ax[abs(i_coil)-1].twinx().plot(time_ppu_data, scan_ppu_data, 'r')

# Visualize with a running mean...
sub_sel_coil = np.arange(-8, 0)
fig, ax = plt.subplots(len(sub_sel_coil))
for i_coil in sub_sel_coil:
    running_mean_data = harray.running_mean(coil_lines[i_coil, center_index], int(n_lines_per_coil * 0.025))
    ax[abs(i_coil)-1].plot(time_coil_data, running_mean_data, 'b')




