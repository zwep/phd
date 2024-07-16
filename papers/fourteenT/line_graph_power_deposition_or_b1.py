import re
import os
import numpy as np
import helper.misc as hmisc
import helper.plot_class as hplotc
import h5py
import matplotlib.pyplot as plt
import scipy.io

"""
This script is mainly a first translation of Bart his script.

We also plot a line graph of the power deposition or the b1

"""

ddata = '/home/bugger/Documents/data/14T'
ddest = '/home/bugger/Documents/paper/14T/plots'
file_list = os.listdir(ddata)
mat_files = [x for x in file_list if x.endswith('mat')]

sel_mat_file = mat_files[0]
print(sel_mat_file)
file_name = sel_mat_file.split('_')[0]
sel_mat_path = os.path.join(ddata, sel_mat_file)
# for sel_mat_file in mat_files:
n_ports = int(re.findall('([0-9]+) Channel', sel_mat_file)[0])

# Global parameters... dit komt van de vertaling van de mat-files
head_weight = 5  # kg
slice_offset = 13  # Wat doet dit precies..?
h5_key = 'ProcessedData'
if '7T' in sel_mat_file:
    sigma_value = 0.413
else:
    sigma_value = 0.5015  # target_mask is selected based on conductivity, gray matter


# Read and load specific data points
with h5py.File(sel_mat_path, 'r') as h5_obj:
    mat_obj = h5_obj[h5_key]
    # Single keys
    sel_key = ['SAR_C', 'aveMass', 'averageSAR', 'calc_VOPs', 'mask_substrate', 'median_filter', 'real_order', 'substrate_sig_value',
               'res', 'vops_method', 'vops_overest_per']
    for i_key in sel_key:
        print(i_key, np.array(mat_obj[i_key]))
    hmisc.print_dict(mat_obj)
    sigma_array = np.array(mat_obj['sigma'])
    b1p_array = np.array(mat_obj['B1'][:, 0]['real'] + 1j * mat_obj['B1'][:, 0]['imag']) * 1e6 #  select B1+, go to muT
    b1m_array = np.array(mat_obj['B1'][:, 1]['real'] + 1j * mat_obj['B1'][:, 1]['imag']) * 1e6
    Ex_flat = np.array(mat_obj['Ex']['real'] + 1j * mat_obj['Ex']['imag']).reshape((n_ports, -1))
    Ey_flat = np.array(mat_obj['Ey']['real'] + 1j * mat_obj['Ey']['imag']).reshape((n_ports, -1))
    Ez_flat = np.array(mat_obj['Ez']['real'] + 1j * mat_obj['Ez']['imag']).reshape((n_ports, -1))
    res_value = float(mat_obj['res'][0][0])
    VOP_array = np.array(mat_obj['VOPm']['real'] + 1j * mat_obj['VOPm']['imag'])

# We loaded the following data + shapes
print("Sigma array ", sigma_array.shape)
print("B1p array ", b1p_array.shape)
print("B1m array ", b1m_array.shape)
print("E_x array ", Ex_flat.shape)
print("E_y array ", Ey_flat.shape)
print("E_z array ", Ez_flat.shape)
print("Res value.. ", res_value)
print("VOP array ", VOP_array.shape)

# Collect parameters from the loaded data...
n_VOP = VOP_array.shape[-1]
sigma_shape = sigma_array.shape
# Create a mask....
sigma_mask = np.zeros(sigma_shape)
sigma_index = np.abs(sigma_array - sigma_value) < 0.001
sigma_mask[sigma_index] = 1

b1p_array_flat = b1p_array.reshape((n_ports, -1))
b1m_array_flat = b1m_array.reshape((n_ports, -1))

"""Calculate Power Depostion Matrix"""
sigma_array_flat = sigma_array.ravel()
power_deposition_matrix = np.zeros((n_ports, n_ports), dtype=complex)
for i in range(n_ports):
    for j in range(i, n_ports):
        print(f"Calculating ... ({i} / {j}) : ({n_ports} / {n_ports})", end='\r')
        E_xx = Ex_flat[i] * Ex_flat[j].conjugate()
        E_yy = Ey_flat[i] * Ey_flat[j].conjugate()
        E_zz = Ez_flat[i] * Ez_flat[j].conjugate()
        # Moet er in principe geen ABS rondom de E_xx + E_yy + E_zz?
        # Nee, dit gebeurt 'automatisch' als i==j. De andere elementen zijn complex
        power_deposition_ij = sum(0.5 * sigma_array_flat * (E_xx + E_yy + E_zz)) * (res_value ** 3)
        power_deposition_matrix[i, j] = power_deposition_ij


i = j = 0
sel_slice = 50
#
E_xx = Ex_flat[i] * Ex_flat[j].conjugate()
E_yy = Ey_flat[i] * Ey_flat[j].conjugate()
E_zz = Ez_flat[i] * Ez_flat[j].conjugate()
temp = 0.5 * sigma_array_flat * (E_xx + E_yy + E_zz)
temp = temp.reshape(sigma_shape)[sel_slice]
sel_b1p = b1p_array[i, sel_slice]
p0 = np.array(np.unravel_index(np.abs(sel_b1p).argmax(), sel_b1p.shape))[::-1]

# p0 = np.array([p0[1], p0[0]])
p_mid = np.array(sigma_shape[1:]) // 2
r = np.sqrt(np.sum((p_mid - p0) ** 2))
rad_thing = np.arccos((p0[0] - p_mid[0]) / (np.linalg.norm(p0 - p_mid)))
# Not sure if this minus sign needs to be there always...
# Probably has to do with the quadrant..
x_range = -np.arange(r) * np.cos(rad_thing) + p0[0]
y_range = np.arange(r) * np.sin(rad_thing) + p0[1]
hplotc.ListPlot([sel_b1p], augm='np.abs')
plt.scatter(x_range, y_range)
plt.scatter(*p_mid)
plt.scatter(*p0)

map_coordinates = np.stack([x_range, y_range])
# Position of the coil..?

import scipy.ndimage
res = scipy.ndimage.map_coordinates(sel_b1p, map_coordinates)
plt.figure()
plt.plot(res_value * np.arange(len(res)), np.abs(res))

