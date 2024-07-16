import numpy as np
import sigpy.mri
import matplotlib.pyplot as plt
import scipy.io
import helper.reconstruction as hrecon
import helper.misc as hmisc
import helper.plot_class as hplotc
import os


if __name__ == "__main__":
    """
    With Barts data we got two acquisitions... for both we have the angles...
    """

    base_dir = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_12_01/bart_data'
    scan_base_dir = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_12_01/ca_29045'
    dangles = os.path.join(base_dir, 'bart_17_2_angles.mat')
    angle_array = scipy.io.loadmat(dangles)['bart_17_2_angles'].reshape(-1)
    sel_sin_file = os.path.join(scan_base_dir, 'ca_01122021_1019026_17_2_transverse_retro_radialV4.sin')
    n_spokes = hrecon.get_spokes_from_sin_file(sel_sin_file)
    acquisition_angles = hrecon.get_angle_spokes(n_spokes)
    # Validate that our angle calculation is correct
    plt.plot(angle_array, 'bo')
    plt.plot(acquisition_angles, 'r*')

    dangles = os.path.join(base_dir, 'bart_14_2_angles.mat')
    angle_array = scipy.io.loadmat(dangles)['bart_14_2_angles'].reshape(-1)
    sel_sin_file = os.path.join(scan_base_dir, 'ca_01122021_1016141_14_2_transverse_retro_radialV4.sin')
    n_spokes = hrecon.get_spokes_from_sin_file(sel_sin_file)
    acquisition_angles = hrecon.get_angle_spokes(n_spokes)
    # Validate that our angle calculation is correct
    plt.plot(angle_array, 'bo')
    plt.plot(acquisition_angles, 'r*')
