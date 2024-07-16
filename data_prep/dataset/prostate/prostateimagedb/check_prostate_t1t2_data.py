
import nrrd
import helper.plot_class as hplotc
import helper.plot_fun as hplotf
import os
import numpy as np
import data_generator.InhomogRemoval as dg_inhom

dir_data_rho = '/home/bugger/Documents/data/prostateimagedatabase/rho'
dir_data_t1 = '/home/bugger/Documents/data/prostateimagedatabase/T1_series'
dir_data_t2 = '/home/bugger/Documents/data/prostateimagedatabase/T2_series'

dir_data_prostate_t1t2 = '/home/bugger/Documents/data/semireal/prostate_simulation_t1t2_rxtx'


"""
Check single files below...
"""

file_list_rho = os.listdir(dir_data_rho)
file_list_t1 = os.listdir(dir_data_t1)
file_list_t2 = os.listdir(dir_data_t2)

i_rho_file = file_list_rho[0]

patient_id = i_rho_file.split('.')[0]
exam_id = i_rho_file.split('.')[1]
series_id = i_rho_file.split('.')[2]

patient_list_t1, exam_list_t1, series_list_t1, ext = zip(*[x.split('.') for x in file_list_t1])
index_patient_t1 = patient_list_t1.index(patient_id)

patient_list_t2, exam_list_t2, series_list_t2, ext = zip(*[x.split('.') for x in file_list_t2])
index_patient_t2 = patient_list_t2.index(patient_id)

sel_slice_rho = 38
sel_slice_t1t2 = 20
file_rho = os.path.join(dir_data_rho, i_rho_file)
rho_array, _ = nrrd.read(file_rho)
hplotc.SlidingPlot(np.moveaxis(rho_array, -1, 0))

fig_single = hplotf.plot_3d_list(np.rot90(np.moveaxis(rho_array, -1, 0)[sel_slice_rho][None], 3, axes=(-2, -1)), ax_off=True)
coords_prostate = {"coordinates0": [74, 186], "coordinates1": [70, 182]}
y0, y1 = coords_prostate['coordinates0']
x0, x1 = coords_prostate['coordinates1']
import matplotlib.pyplot as plt
# plt.imshow(rho_array[:, :, 0])
plt.hlines(y0, x0, x1, colors='r', lw=4)
plt.hlines(y1, x0, x1, colors='r', lw=4)
plt.vlines(x1, y0, y1, colors='r', lw=4)
plt.vlines(x0, y0, y1, colors='r', lw=4)
fig_single.savefig('/home/bugger/Documents/documentatie/Inhomogeneity_removal/abs_data_real_prostate_single.png')

fig = hplotf.plot_3d_list(np.rot90(np.moveaxis(rho_array, -1, 0)[::10][None], 3, axes=(-2, -1)), ax_off=True)
fig.savefig('/home/bugger/Documents/documentatie/Inhomogeneity_removal/abs_data_real_prostate.png')

file_t1 = os.path.join(dir_data_t1, file_list_t1[index_patient_t1])
t1_array, _ = nrrd.read(file_t1)
hplotc.SlidingPlot(np.moveaxis(t1_array, -1, 0))

file_t2 = os.path.join(dir_data_t2, file_list_t2[index_patient_t2])
t2_array, _ = nrrd.read(file_t2)
hplotc.SlidingPlot(np.moveaxis(t2_array, -1, 0))
fig_single = hplotf.plot_3d_list(np.rot90(np.moveaxis(t2_array, -1, 0)[sel_slice_t1t2][None], k=3, axes=(-2,-1)),
                                 ax_off=True, vmin=(0, 2000))
fig_single.savefig('/home/bugger/Documents/documentatie/Inhomogeneity_removal/abs_data_real_t2_prostate_single.png')
fig = hplotf.plot_3d_list(np.rot90(np.rot90(np.rot90(np.moveaxis(t2_array, -1, 0)[::10][None], axes=(-2,-1)), axes=(-2,-1)), axes=(-2,-1)), ax_off=True)
fig.savefig('/home/bugger/Documents/documentatie/Inhomogeneity_removal/abs_data_real_t2_prostate.png')

"""
"""

import nrrd
import helper.plot_class as hplotc
import helper.plot_fun as hplotf
import os
import numpy as np
import data_generator.InhomogRemoval as dg_inhom

dir_data_prostate_t1t2 = '/home/bugger/Documents/data/semireal/prostate_simulation_t1t2_rxtx'


DG = dg_inhom.DataGeneratorInhomogRemovalT1T2(ddata=dir_data_prostate_t1t2, dataset_type='test')
A = DG.__getitem__(0)


hplotf.plot_3d_list(A[1], augm='np.abs')
n_chan, n_y, n_x = A[1].shape
y_center, x_center = (n_y // 2, n_x // 2)
delta_x = int(0.01 * n_y)
z_mean = np.mean(np.abs(A[1])[:, y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x])
b1_plus_scaled = np.sin(np.abs(A[1]) / z_mean * np.pi/2)**3
hplotf.plot_3d_list(b1_plus_scaled, cbar=True, vmin=(-1, 1))

for x in A:
    if x.ndim == 2:
        x = x[np.newaxis]

    hplotf.plot_3d_list(x, augm='np.abs')

"""
Display simulation data
"""
simulation_path = '/home/bugger/Documents/data/simulation/prostate_mri_mrl'
os.listdir(simulation_path)

import data_generator.Rx2Tx as dg_rxtx
A = dg_rxtx.DataSetSurvey2B1_flavio(ddata=simulation_path, input_shape=(8, 256, 256), complex_type='polar')
a, b = A.__getitem__(0)
fig = hplotf.plot_3d_list([a[::2], b[::2]], ax_off=True, subtitle=[['receive'] + ['' for _ in range(7)], ['transmit'] + ['' for _ in range(7)]],
                          vmin=(0, 0.5))
fig.savefig('/home/bugger/Documents/documentatie/Inhomogeneity_removal/abs_data_simulation_prostate.png')
