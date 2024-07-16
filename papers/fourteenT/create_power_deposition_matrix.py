import sys
sys.path.append('/')
import os
import numpy as np
from objective_helper.fourteenT import ReadMatData
from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, DDATA_POWER_DEPOS
import multiprocessing as mp

"""
It always takes so much time to compute these..
Lets precompute it and then do everything again...

So for each coil we can calculate this
"""


file_list = os.listdir(DDATA)
mat_files = [x for x in file_list if x.endswith('mat')]
N = mp.cpu_count()

for i_options in CALC_OPTIONS:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    for sel_mat_file in mat_files:
        mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
        param_container = mat_reader.read_parameters()
        grid_resolution = param_container['resolution']
        mat_container, mask_container = mat_reader.read_mat_object()
        power_deposition_matrix = mat_reader.get_power_deposition_matrix(mat_container, resolution=grid_resolution)
        ddest_power_depost = os.path.join(DDATA_POWER_DEPOS, mat_reader.coil_name)
        np.save(ddest_power_depost, power_deposition_matrix)

