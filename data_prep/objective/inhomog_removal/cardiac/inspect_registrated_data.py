import numpy as np
import os
import helper.plot_class as hplotc
import h5py

"""
It has been some time... lets check what we created in the past..

The B1p/B1m files are fine
The Rho files are from the simulation thing as well and dont represesnt any MR scan of some sorts
"""
simulation_base_dir = '/home/bugger/Documents/data/simulation'

scan_type = '4ch' # Either p2ch or 4ch
sim_plus_dir = f'{simulation_base_dir}/cardiac/{scan_type}/b1_plus'
sim_minus_dir = f'{simulation_base_dir}/cardiac/{scan_type}/b1_minus'
sim_rho_dir = f'{simulation_base_dir}/cardiac/{scan_type}/rho'

for i_file in os.listdir(sim_plus_dir):
    file_b1p = os.path.join(sim_plus_dir, i_file)
    file_b1m = os.path.join(sim_minus_dir, i_file)
    file_rho = os.path.join(sim_rho_dir, i_file)

    b1p_array = np.load(file_b1p)
    b1m_array = np.load(file_b1m)
    rho_array = np.load(file_rho)

    hplotc.ListPlot(b1p_array.sum(axis=0), augm='np.abs')
    hplotc.ListPlot(b1m_array, augm='np.abs')
    hplotc.ListPlot(rho_array, augm='np.abs')

