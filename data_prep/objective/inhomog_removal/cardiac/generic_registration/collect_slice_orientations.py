"""
We want to see how different all the slices are oriented on the SA data

Currently found these avergae slice orientations

(ap, fh, lr)??

# array([ 0.        , 28.13473684, 40.91894737])
# array([36.808, 47.132,  0.   ])

"""

import warnings
import helper.plot_class as hplotc
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import scipy.ndimage
import time
import importlib
import os
import warnings
import pandas as pd
import numpy as np
import reconstruction.ReadCpx as read_cpx
import reconstruction.SenseUnfold as sense_unfold
import re
import data_prep.unfolding_data.ProcessVnumber as proc_vnumber


def get_v_number(path):
    scan_files = {}
    for d, sd, f in os.walk(path):
        regex_vnumber = re.findall('V9_[0-9]*', d)
        if regex_vnumber:
            v_number = regex_vnumber[0]
            scan_files.setdefault(v_number, [])

    return scan_files


scan_dir = '/media/bugger/MyBook/data/7T_scan/cardiac'
target_dir = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac'

vnumber_dict = get_v_number(scan_dir)
# This is how we get all them v-numbers...
unique_v_numbers = list(sorted(vnumber_dict.keys()))
v_number = unique_v_numbers[0]

list_orientations = []
for v_number in unique_v_numbers:
    print('Vnumber ', v_number)
    proc_obj = proc_vnumber.ProcessVnumber(v_number, scan_dir=scan_dir,
                                           target_dir=target_dir, debug=True,
                                           status=False, save_format='npy')
    sel_cine_files = [x for i, x in proc_obj.cine_file_str if 'sa' in x]

    for i_file in sel_cine_files:
        print(i_file)
        cpx_obj = read_cpx.ReadCpx(i_file)
        if cpx_obj.header is not None:
            sense_obj = sense_unfold.SenseUnfold(cpx_obj, pre_loaded_image=np.empty((10, 10, 10)))
            orientation_dict = {'file_name': i_file, 'angulation': sense_obj.angulation_ref,
                                'off_centre': sense_obj.off_centre_ref, 'fov': sense_obj.fov_ref}
            list_orientations.append(orientation_dict)


parameter_file = cpx_obj.get_par_file()
boundary_coord, center_coord = sense_obj.get_coordinate_acq_plane(parameter_file)

avg_orientation = []
for i_orientation in list_orientations:
    print(i_orientation['angulation'])
    if i_orientation['angulation'][2] == 0:
        print(i_orientation['angulation'])
        avg_orientation.append(i_orientation['angulation'])

np.mean(avg_orientation[:-5], axis=0)

# Two types....
# array([ 0.        , 28.13473684, 40.91894737])
# array([36.808, 47.132,  0.   ])