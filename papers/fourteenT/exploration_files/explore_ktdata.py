import os
import numpy as np
import matplotlib.pyplot as plt
import helper.misc as hmisc
from objective_configuration.fourteenT import DDATA_KT_POWER, DPLOT_KT_POWER, COIL_NAME_ORDER, MID_SLICE_OFFSET
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import scipy.io


mat_files = os.listdir(DDATA_KT_POWER)
output_design_files = [x for x in mat_files if x.startswith('output')]
for sel_file in output_design_files:
    split_file_name = sel_file.split('_')
    coil_name = split_file_name[2]
    kt_number = hmisc.get_base_name(split_file_name[3])
    if '5Kt' in kt_number:
        #
        ddest = os.path.join(DPLOT_KT_POWER, coil_name)
        if not os.path.isdir(ddest):
            os.makedirs(ddest)

        ddest_grad_vector = os.path.join(ddest, f'{kt_number}_gradient_vector_3d.png')
        ddest_rf_vector = os.path.join(ddest, f'{kt_number}_rf_amplitude.png')
        ddest_solution_slices = os.path.join(ddest, f'{kt_number}_solution_slices.png')
        # Load the data...
        sel_mat_file = os.path.join(DDATA_KT_POWER, sel_file)
        mat_obj = scipy.io.loadmat(sel_mat_file)['output']

        # Plot names of object..
        hmisc.print_dtype_names(mat_obj)
        grad_X = mat_obj['Gradient_Waveforms_mT'][0][0]['X'][0][0].reshape(-1)
        grad_Y = mat_obj['Gradient_Waveforms_mT'][0][0]['X'][0][0].reshape(-1)
        grad_Z = mat_obj['Gradient_Waveforms_mT'][0][0]['X'][0][0].reshape(-1)
        grad_vector = np.stack([grad_X, grad_Y, grad_Z])
        norm_grad_vector = np.linalg.norm(grad_vector, axis=0)

        # Visualize the direction of the gradients
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.quiver(np.arange(0, len(grad_X)), 0, 0, grad_X, grad_Y, grad_Z, arrow_length_ratio=0)#
        # # What is the time resolution..?
        # ax.set_ylim(grad_Y.min(), grad_Y.max())
        # ax.set_zlim(grad_Z.min(), grad_Z.max())
        # fig.suptitle('Direction of gradient vector')
        # fig.savefig(ddest_grad_vector)

        # Visualize the absolute RF wave form and plot also the norm of the gradient vector
        fig = hplotf.plot_multi_lines(np.abs(mat_obj['RF_Waveforms_mT'][0][0]))
        fig.axes[0].twinx().plot(norm_grad_vector)
        fig.suptitle('Absolute RF waveform and norm of gradient vector')
        fig.savefig(ddest_rf_vector)

        np.abs(np.sum(mat_obj['RF_Waveforms_mT'][0][0]))    # What is deltaT..?
        mat_obj.values
        # pred_solution = mat_obj['Predicted_Solution'][0][0]
        # pred_plot_list = hplotf.get_all_mid_slices(pred_solution, offset=MID_SLICE_OFFSET)
        # fig_obj = hplotc.ListPlot(pred_plot_list, augm='np.abs')
        # fig_obj.figure.savefig(ddest_solution_slices)
        hplotc.close_all()

# Check de debug files
debug_design_files = [x for x in mat_files if x.startswith('debug')]
mask_files = []
for sel_file in sorted(debug_design_files):
    split_file_name = sel_file.split('_')
    coil_name = split_file_name[2]
    kt_number = hmisc.get_base_name(split_file_name[3])
    if '1Kt' in kt_number:
        # Load the data...
        sel_mat_file = os.path.join(DDATA_KT_POWER, sel_file)
        mat_obj = scipy.io.loadmat(sel_mat_file)
        temp_mask = mat_obj['maps']['mask'][0][0]
        mask_files.append(temp_mask)
        # # hmisc.print_dtype_names(mat_obj)
        # for k, v in mat_obj.items():
        #     if k.startswith('__'):
        #         pass
        #     else:
        #         print('\t', k)
        #         hmisc.print_dtype_names(v)
        """
        Wave forms
        """
        rf_full_waveform = mat_obj['waveforms'][0][0]['rffull']
        n_points = rf_full_waveform.shape[0]
        dt = mat_obj['waveforms'][0][0]['dt']
        x_range = np.arange(0, n_points * dt, dt)
        print(f'Duration {coil_name} {kt_number}  {x_range.max().round(5) * 1e3} msec')
        fig = hplotf.plot_multi_lines(np.abs(rf_full_waveform), x_range=x_range)
        """
        Plot of RF 'power'?...
        """
        rf_peak_only = mat_obj['waveforms'][0][0]['rf']
        fig, ax = plt.subplots()
        sel_coil = 0
        ax.plot(np.abs(rf_full_waveform)[:, sel_coil])
        for i_point in rf_peak_only[:, sel_coil]:
            fig.axes[0].hlines(xmin=0, xmax=len(rf_full_waveform), y=np.abs(i_point), color='k')
        """
        Plot of gradients..
        """
        gradient_vector = mat_obj['waveforms'][0][0]['g']
        grad_X = gradient_vector[:, 0]
        grad_Y = gradient_vector[:, 1]
        grad_Z = gradient_vector[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.quiver(np.arange(0, len(grad_X)), 0, 0, grad_X, grad_Y, grad_Z, arrow_length_ratio=0)  #
        ax.set_ylim(grad_Y.min(), grad_Y.max())
        ax.set_zlim(grad_Z.min(), grad_Z.max())
        """
        Visualization of the points in kT space
        """
        kt_points_coords = mat_obj['waveforms'][0][0]['k']
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*kt_points_coords.T)
        # This is simply the time resolution
        dt = mat_obj['waveforms'][0][0]['dt']
        # This is simply the norm of the location of the kT points
        kspace_dist = mat_obj['waveforms'][0][0]['k_dist']
        """
        Exploring the content of maps..
        """
        b0_array = mat_obj['maps'][0][0]['b0']
        mask_array = mat_obj['maps'][0][0]['mask']
        b1_array = mat_obj['maps'][0][0]['b1']
        phsinit_array = mat_obj['maps'][0][0]['phsinit']

        mask_list = hplotf.get_all_mid_slices(mask_array)
        b0_list = hplotf.get_all_mid_slices(b0_array)
        b1_list = hplotf.get_all_mid_slices(b1_array)
        phsinit_list = hplotf.get_all_mid_slices(phsinit_array)
        hplotc.ListPlot(mask_list + b0_list + phsinit_list, augm='np.abs')
        hplotc.ListPlot(b1_list, augm='np.abs')
        """
        Algorithm options..
        """
        print('nthreads', mat_obj['algp'][0][0]['nthreads'])
        print('compute method', mat_obj['algp'][0][0]['computemethod'])
        """
        Exploring the options in .. prbp
        """
        name_list = mat_obj['prbp'][0][0].dtype.names
        space_str = 15
        for i_name in name_list:
            if i_name in ['xx', 'yy', 'zz']:
                pass
            else:
                len_name = len(i_name)
                print(i_name, ' ' *(space_str - len_name), mat_obj['prbp'][0][0][i_name][0][0])

        XX = mat_obj['prbp'][0][0]['xx'][0][0]
        YY = mat_obj['prbp'][0][0]['yy'][0][0]
        ZZ = mat_obj['prbp'][0][0]['zz'][0][0]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(XX, YY, ZZ)


""" Check the beta adjust files..."""
# Check de debug files
from objective_configuration.fourteenT import DDATA_KT_BETA_POWER

mat_files = os.listdir(DDATA_KT_BETA_POWER)
output_files = [x for x in mat_files if x.startswith('output')]
output_files = sorted(output_files, key=lambda x: int(re.findall('_([0-9]*)beta', x)[0]))

axial_list = []
for sel_file in output_files:
    split_file_name = sel_file.split('_')
    coil_name = split_file_name[2]
    kt_number = hmisc.get_base_name(split_file_name[3])
    # Load the data...
    sel_mat_file = os.path.join(DDATA_KT_BETA_POWER, sel_file)
    mat_obj = scipy.io.loadmat(sel_mat_file)
    hmisc.print_dtype_names(mat_obj['output'])
    kt_array = mat_obj['output']['Predicted_Solution'][0][0].T
    mid_slices = hplotf.get_all_mid_slices(kt_array)[-1]
    axial_list.append(mid_slices)

hplotc.SlidingPlot(np.array(axial_list))

import helper.metric as hmetric
plt.plot([hmetric.coefficient_of_variation(np.abs(x[x>0])) for x in axial_list])


""" Check the beta files..."""
# Check de debug files
from objective_configuration.fourteenT import DDATA_KT_BETA_VOP

mat_files = os.listdir(DDATA_KT_BETA_VOP)
for sel_coil in COIL_NAME_ORDER:
    output_files = [x for x in mat_files if x.startswith('output') and sel_coil in x]
    output_files = sorted(output_files, key=lambda x: float(re.findall('_([0-9]*\.[0-9]*)beta', x)[0]))

    debug_files = [x for x in mat_files if x.startswith('debug') and sel_coil in x]
    debug_files = sorted(debug_files, key=lambda x: float(re.findall('_([0-9]*\.[0-9]*)beta', x)[0]))
    axial_list = []
    for sel_file in debug_files:
        split_file_name = sel_file.split('_')
        coil_name = split_file_name[2]
        kt_number = hmisc.get_base_name(split_file_name[3])
        # Load the data...
        sel_mat_file = os.path.join(DDATA_KT_BETA_VOP, sel_file)
        mat_obj = scipy.io.loadmat(sel_mat_file)
        print(mat_obj['waveforms'][0][0]['dt'][0][0])

        hmisc.print_dtype_names(mat_obj['output'])
        kt_array = mat_obj['output']['Predicted_Solution'][0][0].T
        mid_slices = hplotf.get_all_mid_slices(kt_array)[-1]
        axial_list.append(mid_slices)

    hplotc.SlidingPlot(np.array(axial_list))

import helper.metric as hmetric
plt.plot([hmetric.coefficient_of_variation(np.abs(x[x>0])) for x in axial_list])


""" Check the VOP files... """
# Check de debug files
from objective_configuration.fourteenT import DDATA_KT_VOP
import re
mat_files = os.listdir(DDATA_KT_VOP)
sel_coil = '16 Channel Loop Dipole Array'
output_files = [x for x in mat_files if x.startswith('output') and sel_coil in x]

output_files = sorted(output_files, key=lambda x: float(re.findall('_([0-9]*)Kt_vop', x)[0]))

axial_list = []
for sel_file in output_files:
    split_file_name = sel_file.split('_')
    coil_name = split_file_name[2]
    kt_number = hmisc.get_base_name(split_file_name[3])
    # Load the data...
    sel_mat_file = os.path.join(DDATA_KT_VOP, sel_file)
    mat_obj = scipy.io.loadmat(sel_mat_file)
    hmisc.print_dtype_names(mat_obj['output'])
    kt_array = mat_obj['output']['Predicted_Solution'][0][0].T
    mid_slices = hplotf.get_all_mid_slices(kt_array)[-1]
    axial_list.append(mid_slices)

hplotc.SlidingPlot(np.array(axial_list))

import helper.metric as hmetric
plt.plot([hmetric.coefficient_of_variation(np.abs(x[x>0])) for x in axial_list])

