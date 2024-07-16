import numpy as np
import helper.array_transf as harray
import sigpy.mri
import matplotlib.pyplot as plt
import scipy.io
import helper.misc as hmisc
import helper.plot_class as hplotc
import os
import helper.reconstruction as hrecon

# Possible parameters for spokes
# SOMEHOW: retro_max_intp_length

"""
Het is zo gek dat retro_max_intp_length overeenkomt met het aantal spokes... dat KAN toch niet.
"""

dbase = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_12_01/bart_data'
dbase_scan = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_12_01/ca_29045'

# Get sorted array
dsorted = os.path.join(dbase, 'bart_17_2_sorted.mat')
sorted_array = scipy.io.loadmat(dsorted)['bart_17_2_sorted']

# Get non-sorted array
dunsorted = os.path.join(dbase, 'bart_17_2_unsorted.mat')
unsorted_array = scipy.io.loadmat(dunsorted)['bart_17_2_unsorted']
# Dit laat wel zien dat het echt om 24 coils gaat.
# De laatste 8 vangen ook duidelijk het meeste signaal
hplotc.ListPlot(unsorted_array[:, :30], augm='np.abs', vmin=(0, 1000), aspect='auto')

sel_sin_file = os.path.join(dbase_scan, 'ca_01122021_1019026_17_2_transverse_retro_radialV4.sin')
min_encoding = hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_min_encoding_nrs').split()
max_encoding = hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_max_encoding_nrs').split()
min_encoding_x = int(min_encoding[0])
max_encoding_x = int(max_encoding[0])

min_encoding_y = int(min_encoding[1])
max_encoding_y = int(max_encoding[1])

n_spokes = max_encoding_y - min_encoding_y + 1
trajectory = hrecon.get_trajectory_n_spoke(min_encoding, max_encoding)

n_coil = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_channel_names'))
n_card = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_cardiac_phases'))
TR = float(hrecon.get_key_from_sin_file(sel_sin_file, 'repetition_times')) * 10 ** -3  # in seconds
ovs = float(hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_grid_overs_factor'))
width = int(hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_grid_kernel_size'))

result = []
for i_coil in range(n_coil):
    temp = []
    for i_card in range(n_card):
        selected_data = unsorted_array[:, i_coil::n_coil][:, i_card::n_card][:, :n_spokes]
        selected_data = np.moveaxis(selected_data, -1, 0)
        temp_img = sigpy.nufft_adjoint(selected_data, coord=trajectory, oversamp=ovs, width=width)
        temp.append(temp_img)
    result.append(temp)

result = np.array(result)


"""
Mathijs data
"""

dbase_scan = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_11_24/V9_28674'
dbase = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_11_24/mathijs_data'

# Get sorted array
dsorted = os.path.join(dbase, 'recon_sorted.mat')
sorted_array = scipy.io.loadmat(dsorted)['recon_sorted'][0][0]
sorted_array.shape

# Get non-sorted array
dunsorted = os.path.join(dbase, 'recon_data.mat')
unsorted_array = scipy.io.loadmat(dunsorted)['recon_data'][0][0]
unsorted_array.shape



sel_sin_file = os.path.join(dbase_scan, 'v9_24112021_1650264_3_2_transradialfast_retroV4.sin')
n_spokes = int(hrecon.get_key_from_sin_file(sel_sin_file, 'retro_max_intp_length'))
calc_trajectory = hrecon.get_trajectory_sin_file(sel_sin_file)


# CHeck how many spokes these things have....??
dd = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_06/V9_16935/v9_06022021_1130242_8_3_transradialfast_high_timeV4.cpx'
import reconstruction.ReadCpx as read_cpx
obj = read_cpx.ReadCpx(dd)
temp_A = obj.get_cpx_img()
hplotc.SlidingPlot(temp_A)

sel_sin_file = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_06/V9_16936/v9_06022021_1223187_6_2_transradialfastV4.sin'
min_encoding = hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_min_encoding_nrs').split()
max_encoding = hrecon.get_key_from_sin_file(sel_sin_file, 'non_cart_max_encoding_nrs').split()
min_encoding_x = int(min_encoding[0])
max_encoding_x = int(max_encoding[0])

min_encoding_y = int(min_encoding[1])
max_encoding_y = int(max_encoding[1])

n_enc_y = max_encoding_y - min_encoding_y + 1
n_coil = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_channel_names'))
n_card = int(hrecon.get_key_from_sin_file(sel_sin_file, 'nr_cardiac_phases'))
TR = float(hrecon.get_key_from_sin_file(sel_sin_file, 'repetition_times')) * 10 ** -3 # in seconds
# Deze berekening klopt
# Nu kijken of het aantal spokes ook klopt met wat dinges zegt..?
n_enc_y * n_card * TR
# maarja.. dan mist er toch echt wat data..
