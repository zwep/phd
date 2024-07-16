"""
We are going to visualize some images that we use for the fully sampled radia lstuff
and the cartesian one

These need to be alligned in a way...
"""

import os
import numpy as np
import helper.plot_class as hplotc


dd_cart = '/data/seb/unfolded_radial/cartesian_radial_dataset_4ch/test'


dd_target = '/data/seb'

for sel_dir in [dd_cart]:
    temp_dir = os.path.basename(sel_dir)
    for i_file in os.listdir(os.path.join(dd_cart, 'input')):
        file_name, _ = os.path.splitext(i_file)
        input_file = os.path.join(sel_dir, 'input', i_file)
        target_file = os.path.join(sel_dir, 'target', i_file)
        dest_file = os.path.join(dd_target, file_name + '.png')
        input_array = np.load(input_file)
        target_array = np.load(target_file)

        if input_array.ndim == 4:
            input_array = np.abs(input_array[-8:]).sum(axis=0)
        input_array = np.abs(input_array[0])

        if target_array.ndim == 4:
            target_array = np.abs(target_array[-8:]).sum(axis=0)
        target_array = np.abs(target_array[0])

        plot_obj = hplotc.ListPlot([input_array, target_array], title=i_file)
        plot_obj.figure.savefig(dest_file)
        hplotc.close_all()
