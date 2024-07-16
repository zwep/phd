
import re
import os
import scipy.io
import numpy as np
import helper.plot_class as hplotc
import matplotlib.pyplot as plt
import scipy.io
import os
import helper.reconstruction as hrecon

"""
Here we validate that counting the number of spokes and the given TR gives a reasonable time..

-- This was done WITHOUT the retrospective patch..
"""

ddata = '/media/bugger/MyBook/data/7T_data/cardiac_unsorted_data/V9_19531/unsorted_data'
file_stuff = [os.path.join(ddata, x) for x in os.listdir(ddata)]
sel_file = file_stuff[0]
unsorted_radial = scipy.io.loadmat(sel_file)['reconstructed_data']
# This looks reasonable (This is for the transradial_fast)
# number of acquisition X TR / n_coils
# We get like 8 seconds... which is OK I guess
# 50688 * 4*10**-3 / 24

sel_file = file_stuff[1]
unsorted_radial_high_time = scipy.io.loadmat(sel_file)['reconstructed_data']
# This looks reasonable (This is for the transradial_fast_high_time)
# number of acquisition X TR / n_coils
# We get like 8 seconds... which is OK I guess
# Similar... but different
# 46800 * 4*10**-3 / 24

sel_file = file_stuff[2]
untrig_radial_high_time = scipy.io.loadmat(sel_file)['reconstructed_data']
# Also this looks OK. This is the untriggered case
# We get something like 20 seconds of data.. which is OK.
# 126720 * 4*10**-3 / 24

"""
This was done WITH the retrospective patch

-- Here we see that we dont get the spokes that we want...
"""

ddata = '/media/bugger/MyBook/data/7T_scan/phantom/2021_12_09/mat_data'
list_files = [os.path.join(ddata, x) for x in os.listdir(ddata)]
unsorted_list = []
for x in list_files:
    A = scipy.io.loadmat(x)['unsorted_data']
    n_spokes = A.shape[1]
    file_name, _ = os.path.splitext(x)
    print(file_name)
    sin_file = file_name + '.sin'
    sin_file = re.sub('mat_data', 'ca_29447', sin_file)
    if os.path.isfile(sin_file):
        TR = float(hrecon.get_key_from_sin_file(sin_file, 'repetition_times'))
        n_coils = int(hrecon.get_key_from_sin_file(sin_file, 'nr_measured_channels'))
        print("Total time ", n_spokes * (TR * 10 ** -3) / n_coils)
    else:
        print("No sin file found... estimate: ", n_spokes * (TR * 10 ** -3) / n_coils)

"""
Here we check another fantom file...
Not sure which acquisition this was exactly.. from the other .sin files I get that
the number of coils was 8... 
"""

other_mat_file = '/media/bugger/MyBook/data/7T_scan/phantom/2021_10_20/fantom_data.mat'
possible_sin_file = '/media/bugger/MyBook/data/7T_scan/phantom/2021_10_20/se_26726/se_20102021_1912369_27_2_surveylr_5_phasesV4.sin'
TR = float(hrecon.get_key_from_sin_file(possible_sin_file, 'repetition_times'))
n_coils = int(hrecon.get_key_from_sin_file(possible_sin_file, 'nr_measured_channels'))
scipy_obj = scipy.io.loadmat(other_mat_file)
n_spokes = scipy_obj['unsorted_data'].shape[1]
print("Total time ", n_spokes * (TR * 10 ** -3) / n_coils)

# This should be the time and the number of spokes
total_nr_profiles = int(hrecon.get_key_from_sin_file(possible_sin_file, 'total_nr_profiles'))
print("Total time ", total_nr_profiles * (TR * 10 ** -3) / n_coils)