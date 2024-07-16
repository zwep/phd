import sys
sys.path.append('/')
import numpy as np
import json
import helper.misc as hmisc
import os
import helper.plot_class as hplotc
from objective_helper.fourteenT import ReadMatData, OptimizeData
from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, SUBDIR_RANDOM_SHIM, RF_SCALING_FACTOR
import multiprocessing as mp

"""
Here we calculate random B1 shims normalized to the head SAR limit I believe..

We use parallel or loops in this acse...
"""

file_list = os.listdir(DDATA)
mat_files = [x for x in file_list if x.endswith('mat')]
N = mp.cpu_count()
n_shims = 9000


for i_options in CALC_OPTIONS:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    print(mat_files)
    for sel_mat_file in mat_files:
        mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
        # mat_reader.print_content_mat_obj()
        dest_path = os.path.join(ddest, SUBDIR_RANDOM_SHIM, mat_reader.coil_name)
        if not os.path.isdir(dest_path):
            os.makedirs(dest_path)

        ddest_dict = os.path.join(dest_path, 'random_shim.json')
        # CHeck if the json file already exists... and if so how long it is.
        # If the size of the random shims is >10.000, then we are done
        if os.path.isfile(ddest_dict):
            temp_json = hmisc.load_json(ddest_dict)
            n_random_shims = len(temp_json['random_shim'])
        else:
            n_random_shims = 0

        if n_random_shims >= n_shims:
            print('The current shim file already contains enough shims? Continue anyway.', sel_mat_file)
        else:
            print(sel_mat_file)

        data_obj = OptimizeData(ddest=ddest, mat_reader=mat_reader, full_mask=full_mask, type_mask=type_mask)

        def _temp_mp_fun(x):
            # X is needed here for the MP process
            temp = data_obj.get_binned_random_shim(x)
            temp_dict = data_obj.get_result_container(temp)
            return temp_dict

        with mp.Pool(processes=N//2) as p:
            list_random_shim_results = p.map(_temp_mp_fun, list(range(n_shims)))

        result_dict_list = hmisc.listdict2dictlist(list_random_shim_results)
        result_dict_list['opt_shim'] = [[list(x.real), list(x.imag)] for x in result_dict_list['opt_shim']]
        result_dict_list['random_shim'] = result_dict_list['opt_shim']
        del result_dict_list['opt_shim']

        json_ser_obj = json.dumps(result_dict_list)
        with open(ddest_dict, 'w') as f:
            f.write(json_ser_obj)

        hplotc.close_all()
        del mat_reader
        del data_obj
