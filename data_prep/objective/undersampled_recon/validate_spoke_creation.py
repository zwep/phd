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


if __name__ == "__main__":
    """
    With Barts data we got two acquisitions... for both we have the angles...
    """

    # First the file 17..
    dbase = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_12_01/bart_data'
    dbase_scan = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_12_01/ca_29045'
    dkpos = os.path.join(dbase, 'bart_17_2_kpos.mat')
    kpos_array = scipy.io.loadmat(dkpos)['bart_17_2_kpos']
    sel_sin_file = os.path.join(dbase_scan, 'ca_01122021_1019026_17_2_transverse_retro_radialV4.sin')
    n_spokes = int(hrecon.get_key_from_sin_file(sel_sin_file, 'retro_max_intp_length'))
    calc_trajectory = hrecon.get_trajectory_sin_file(sel_sin_file)

    fig, ax = plt.subplots()
    for i in range(n_spokes):
        ax.scatter(kpos_array[:, i, :, 0], kpos_array[:, i, :, 1], color='b', marker='o')
        ax.scatter(calc_trajectory[i, :, 0], calc_trajectory[i, :, 1], color='r', marker='*')

    # Now test number 14...
    dkpos = '/media/bugger/WORK_USB/bart_data/bart_14_2_kpos.mat'
    kpos_array = scipy.io.loadmat(dkpos)['bart_14_2_kpos']
    sel_sin_file = '/media/bugger/WORK_USB/2021_12_01/ca_29045/ca_01122021_1016141_14_2_transverse_retro_radialV4.sin'
    n_spokes = hrecon.get_spokes_from_sin_file(sel_sin_file)
    calc_trajectory = hrecon.get_trajectory_sin_file(sel_sin_file)

    fig, ax = plt.subplots()
    for i in range(n_spokes):
        ax.scatter(kpos_array[:, i, :, 0], kpos_array[:, i, :, 1], color='b', marker='o')
        ax.scatter(calc_trajectory[i, :, 0], calc_trajectory[i, :, 1], color='r', marker='*')