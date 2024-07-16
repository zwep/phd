import sys
sys.path.append('/')
import helper.misc as hmisc
import os
from objective_helper.fourteenT import ReadMatData, OptimizeData, VisualizeAllMetrics
from objective_configuration.fourteenT import CALC_OPTIONS, DDATA

"""
Here we re-calculate the SAR distirbution. Now in a normalized fashion...
"""

file_list = os.listdir(DDATA)
mat_files = [x for x in file_list if x.endswith('mat')]
"""
Here we recalculate everything for the optimal shims...
"""

for i_options in CALC_OPTIONS:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    # Get the metric results from all coil files
    ii = 0
    # print(f"Dealing with opt shim {ii} / 25")
    opt_shim_str = f'opt_shim_{str(ii).zfill(2)}'
    visual_obj = VisualizeAllMetrics(DDATA, ddest=ddest, opt_shim_str=opt_shim_str)

    temp_df = None
    # Loop over the coil files and replace all the non normalized metrics wtih normalized ones
    for sel_mat_file in mat_files:
        print(sel_mat_file)
        mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
        data_obj_norm = OptimizeData(ddest=ddest, mat_reader=mat_reader, full_mask=full_mask, type_mask=type_mask, normalization=True)
        optim_result_dict = visual_obj.optimal_json_data[mat_reader.coil_name]
        new_result_dict = data_obj_norm.get_result_container(optim_result_dict['opt_shim'])
        print(hmisc.print_dict(new_result_dict))
        # print(sel_mat_file, new_result_dict['b1_rms'])


        result_dict = visual_obj.optimized_json_data[mat_reader.coil_name]

        temp_list = []
        for i_shim in result_dict['opt_shim']:
            temp_result_dict = data_obj_norm.get_result_container(i_shim)