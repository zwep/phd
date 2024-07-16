"""
We are going to replicate the result of a paper
"""

import re
import sigpy
import sigpy.mri

import helper.array_transf as harray
import reconstruction.ReadCpx as read_cpx
import os
import numpy as np
import helper.plot_class as hplotc

ddata = '/media/bugger/MyBook/data/7T_scan/phantom/2020_02_19/ph_8653'
file_list = os.listdir(ddata)
sorted_file_list = sorted([x for x in file_list if x.endswith('cpx') and 'radial' in x])
acq_array = []
for i_file in sorted_file_list:
    sel_file = os.path.join(ddata, i_file)
    cpx_obj = read_cpx.ReadCpx(sel_file)
    cpx_array = cpx_obj.get_cpx_img()
    temp_array = np.squeeze(np.abs(cpx_array).sum(axis=0))[1]
    acq_array.append(temp_array)


undersampling_factors = [int(re.findall('acq([0-9]+)V4', x)[0]) for x in sorted_file_list]

"""
Get synthetically undersampled image...
"""

fully_sampled_radial = np.copy(acq_array[0])
fully_sampled_radial = harray.scale_minmax(fully_sampled_radial)

img_shape = fully_sampled_radial.shape[-2:]

n_points = max(img_shape)
n_spokes = int(np.pi / 2 * max(img_shape))
width = 6
ovs = 1.25
synthetically_undersampled = []
for p_undersample in undersampling_factors:
    print(p_undersample)
    # Define undersampled trajectory, the same for ALL the coils
    # Create a new trajectory...
    trajectory_radial = sigpy.mri.radial(coord_shape=(n_spokes, n_points, 2), img_shape=img_shape)
    trajectory_radial = trajectory_radial.reshape(-1, 2)
    dcf = np.sqrt(trajectory_radial[:, 0] ** 2 + trajectory_radial[:, 1] ** 2)

    n_undersample = int((p_undersample / 100) * n_spokes)
    undersampled_trajectory = np.array(np.split(trajectory_radial, n_spokes))
    # random_lines = np.random.choice(range(n_spokes), size=(n_spokes - n_undersample), replace=False)
    random_lines = np.array(list(range(0, n_spokes, int(100 / p_undersample))))
    for ii in range(n_spokes):
        if ii in random_lines:
            continue
        else:
            undersampled_trajectory[ii] = None

    undersampled_trajectory = undersampled_trajectory.reshape(-1, 2)

    temp_kspace = sigpy.nufft(fully_sampled_radial, coord=undersampled_trajectory, width=width, oversamp=ovs)
    temp_img = sigpy.nufft_adjoint(temp_kspace * dcf, coord=undersampled_trajectory, oshape=img_shape, width=width, oversamp=ovs)
    synthetically_undersampled.append(temp_img)

hplotc.ListPlot([synthetically_undersampled, acq_array], augm='np.abs', subtitle=[['synth', '', '', '', '', '', ''],
                                                                                  ['measured', '', '', '', '', '', '']])

def metric_us_artifacts(x, y):
    # Characterizing Radial Undersampling Artifacts for Cardiac Applications
    rmse = np.sqrt(np.sum((x - y) ** 2))
    power = np.sqrt(np.sum(x ** 2))
    return 100 * rmse / power


mask_fantom = harray.get_treshold_label_mask(fully_sampled_radial)
import matplotlib.pyplot as plt
synthetic_artifact_metric = []
acquired_artifact_metric = []
for synth_x, acq_y in zip(synthetically_undersampled, acq_array):
    X = harray.scale_minmax(np.abs(synth_x)) * mask_fantom
    Y = harray.scale_minmax(np.abs(acq_y)) * mask_fantom
    synthetic_metric = metric_us_artifacts(fully_sampled_radial * mask_fantom, X)
    acq_metric = metric_us_artifacts(fully_sampled_radial * mask_fantom, Y)
    synthetic_artifact_metric.append(synthetic_metric)
    acquired_artifact_metric.append(acq_metric)

fig, ax = plt.subplots()
ax.plot(undersampling_factors[::-1], synthetic_artifact_metric[::-1], 'o-',color='r', label='synthetic')
ax.plot(undersampling_factors[::-1], acquired_artifact_metric[::-1], 'o-', color='k', label='measured')
plt.ylim(0, 30)
plt.ylabel('Artifact error')
plt.xlabel('Sampling percentage')
plt.legend()
