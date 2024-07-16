

import numpy as np
import os
import helper.plot_fun as hplotf
import helper.array_transf as harray


"""
Simple Function to check mask creation..

"""

sel_dir = '/home/bugger/Documents/data/simulation/cardiac/b1/p2ch/b1_minus'
dir_b1_minus = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/non_registered/b1minus/sel_filtered_aligned'
dir_b1_plus = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/non_registered/b1plus'
dir_rho = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/non_registered/rho'
dir_b1_minus_axial = '/home/bugger/Documents/data/simulation/cardiac/b1/axial/b1_minus'
dir_b1_minus_p2ch = '/home/bugger/Documents/data/simulation/cardiac/b1/p2ch/b1_minus'
dir_b1_minus_p4ch = '/home/bugger/Documents/data/simulation/cardiac/b1/p4ch/b1_minus'

derp_dirs = [dir_b1_minus, dir_b1_plus, dir_rho, dir_b1_minus_axial, dir_b1_minus_p2ch, dir_b1_minus_p4ch]

for sel_dir in derp_dirs:
    n_files = 1
    treshold_smooth = 0.6
    kernel_factor = 0.2
    plot_intermediate = False
    debug_intermediate = False
    temp_files = os.listdir(sel_dir)
    for i_file in temp_files[:n_files]:
        file_path = os.path.join(sel_dir, i_file)
        temp_array = np.load(file_path)

        print('Loaded array shape ', temp_array.shape)

        ndim = temp_array.ndim
        if ndim == 4:
            sel_slice = temp_array.shape[1]//2
            temp_array = temp_array[:, sel_slice]
        elif ndim == 2:
            temp_array = temp_array[np.newaxis]

        print('Loaded array shape ', temp_array.shape)

        n_kernel = int(min(temp_array.shape[-2:]) * kernel_factor)
        temp_mask = harray.get_smoothed_mask(np.abs(temp_array).sum(axis=0), treshold_smooth=treshold_smooth,
                                                 n_mask=n_kernel, debug=debug_intermediate,
                                                 conv_boundary='fill',
                                                 conv_mode='valid')
        if debug_intermediate:
            temp_plot_array = []
            for k, v in temp_mask.items():
                temp_plot_array.append(v)
            hplotf.plot_3d_list(temp_plot_array)

        hplotf.plot_3d_list([temp_mask, np.abs(temp_array).sum(axis=0)], title=sel_dir)
