import helper.reconstruction as hrecon
import helper.plot_class as hplotc
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import reconstruction.ReadPhyslog as read_phys
import reconstruction.RetrospectiveCardiac as retro_cardiac

"""
Okay.. now that we see that we might have the cardiac movement.....

Can we use this data to sort stuff???

Averaging on the acquired data center points did not prove to be that useful..

"""
#
# dmatthijs = '/media/bugger/B7DF-4571/2022_01_19/mat_data'
# unsorted_100p_data = 'ca_19012022_1710203_10_2_transradialdyn_20V4.mat'
# unsorted_100p_sin = 'ca_19012022_1710203_10_2_transradialdyn_20V4.sin'
# dphyslog = 'SCANPHYSLOG_ca_19012022_1710203_10_1_transradialdyn_20V4.log'
#
# import importlib
# importlib.reload(read_phys)
# phys_obj_100p = read_phys.ReadPhyslog(os.path.join(dmatthijs, dphyslog))
# phys_obj_100p.visualize_label('ppu')
#
#
# # Read the data
# mat_file = os.path.join(dmatthijs, unsorted_100p_data)
# sin_file = os.path.join(dmatthijs, unsorted_100p_sin)
#
# signal_obj = retro_cardiac.GetCardiacSignal(mat_file, sin_file)
# signal_obj.visualize_averaged_center_signal()
# cardiac_signal = signal_obj.get_averaged_center_signal(sel_coil=-2, complex_part='imag')
#
# binning_obj = retro_cardiac.SignalBinning(cardiac_signal, signal_obj.TR)
# binning_obj.visualize_cardiac_bins()
#
# sort_obj = retro_cardiac.SortCardiacRadialData(unsorted_radial_data=mat_file, sin_file=sin_file, cardiac_signal=cardiac_signal,
#                                  n_cardiac_phases=30, sampling_freq=signal_obj.TR, golden_angle=True)
#
#
# for i in range(sort_obj.n_cardiac_phases):
#     sort_obj.visualize_trajectory(i)
# # Visualize the trajecotyr..
#
#
# reconstructed_image = sort_obj.reconstruct_cardiac_cine()
#
# hplotc.SlidingPlot(reconstructed_image.sum(axis=1))
#
# # Lets store it...
# ddest = '/media/bugger/MyBook/data/7T_data/sorted_untriggered_data'
# np.save(os.path.join(ddest, 'matthijs_20p_untriggered.npy'), reconstructed_image)
#
# dload = '/media/bugger/MyBook/data/7T_data/sorted_untriggered_data/matthijs_20p_untriggered.npy'
# A = np.load(dload)
# zz = np.abs(A).sum(axis=1)
# hplotc.SlidingPlot(zz)
#
# n_spokes = 100
#
# angle_offset = np.pi
# angle_factor = np.pi / n_spokes
# ppe_definition = np.array([angle_ix * angle_factor + angle_offset % (2 * np.pi) for angle_ix in range(n_spokes)])
#
# fig, ax = plt.subplots()
# ax.plot(hrecon.get_angle_spokes(n_spokes), 'b*')
# ax.plot(ppe_definition, 'r*')
#
# # Check with sin/cos stuff..
# plt.figure()
# plt.scatter(np.cos(ppe_definition), np.sin(ppe_definition))
# plt.scatter(-np.cos(ppe_definition), -np.sin(ppe_definition))


"""
Now try it on Weezel's data
"""


# dsin = '/media/bugger/MyBook/data/7T_scan/cardiac/2022_03_05/V9_32575//v9_05032022_1032049_8_1_transverse_radial_retrospectiveV4.sin'
# dmat = '/media/bugger/MyBook/data/7T_scan/cardiac/2022_03_05/V9_32575/matdata/v9_05032022_1032049_8_1_transverse_radial_retrospectiveV4.lab_data.mat'
dsin = '/media/bugger/MyBook/data/7T_scan/cardiac/2022_03_05/V9_32575//v9_05032022_1032049_8_1_transverse_radial_retrospectiveV4.sin'
dmat = '/media/bugger/MyBook/data/7T_scan/cardiac/2022_03_05/V9_32575/matdata/v9_05032022_1032049_8_1_transverse_radial_retrospectiveV4.lab_data.mat'
signal_obj = retro_cardiac.GetCardiacSignal(dmat, dsin)
n_coil = int(hrecon.get_key_from_sin_file(dsin, 'nr_channel_names'))
signal_obj.visualize_averaged_center_signal()

cardiac_signal = signal_obj.get_averaged_center_signal(sel_coil=-2, complex_part='imag')
plt.plot(cardiac_signal)
binning_obj = retro_cardiac.SignalBinning(cardiac_signal, signal_obj.TR, distance=0.9 * 1/ signal_obj.TR)
binning_obj.visualize_cardiac_bins()

sort_obj = retro_cardiac.SortCardiacRadialData(unsorted_file=dmat, sin_file=dsin, cardiac_signal=cardiac_signal,
                                               n_cardiac_phases=30, sampling_freq=signal_obj.TR, golden_angle=True)

for i in range(sort_obj.n_cardiac_phases):
    sort_obj.visualize_trajectory(i)

reconstructed_image = sort_obj.reconstruct_cardiac_cine()
res = reconstructed_image.sum(axis=1)

hplotc.SlidingPlot(res)
