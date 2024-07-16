from objective_helper.fourteenT import VisualizeAllMetrics, DataCollector
from objective_helper.fourteenT import VisualizeAllMetrics, DataCollector, ReadMatData
import matplotlib.pyplot as plt
import scipy.io
import helper.misc as hmisc
import os
import objective_helper.fourteenT as helper_14T
from objective_configuration.fourteenT import SUBDIR_OPTIM_SHIM, COIL_NAME_ORDER, CALC_OPTIONS, DDATA, COLOR_DICT, \
    DPLOT_1KT_BETA_POWER, DDATA_1KT_BETA_POWER, SUBDIR_OPTIM_SHIM, \
    DPLOT_1KT_BETA_VOP, DDATA_1KT_BETA_VOP, \
    DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP, \
    DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER, \
    COIL_NAME_ORDER_TRANSLATOR, RF_SCALING_FACTOR_1KT, \
    RF_SCALING_FACTOR, WEIRD_RF_FACTOR, TARGET_FLIP_ANGLE, SUBDIR_RANDOM_SHIM
import os
import json
import numpy as np

"""
Lets recalculate stuff... and simpify it..
We want the shim.. and the peak SAR and B1 nrmse. Thats it
We will scale the rest
"""

file_list = os.listdir(DDATA)
mat_files = [x for x in file_list if x.endswith('mat')]

# Filter on SAR results...
for i_options in CALC_OPTIONS:
    full_mask = i_options['full_mask']
    type_mask = i_options['type_mask']
    ddest = i_options['ddest']
    for sel_mat_file in mat_files:
        mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
        Q_container = mat_reader.read_Q_object()['Q10g']
        data_obj = DataCollector(mat_reader, full_mask=full_mask, type_mask=type_mask)
        n_results = min([len(os.listdir(os.path.join(ddest, 'optim_shim', x))) for x in COIL_NAME_ORDER])
        temp_list = []
        for ii in range(n_results):
            visual_obj = VisualizeAllMetrics(ddest=ddest, opt_shim_str=f'opt_shim_{str(ii).zfill(2)}')
            temp_result_dict = visual_obj.optimized_json_data[mat_reader.coil_name]
            #
            json_dest_dir = os.path.join(visual_obj.ddest, 'optim_shim_recalc_sar', mat_reader.coil_name)
            if not os.path.isdir(json_dest_dir):
                os.makedirs(json_dest_dir)
            #
            result = []
            for temp_shim in temp_result_dict['opt_shim']:
                temp_dict = {}
                rmse, nrmse, avg_b1, rmse_normalized, nrmse_normalized, avg_b1_normalized = data_obj.get_nrmse_b1p(x_shim=temp_shim)
                # peakSAR, peakSAR_normalized, optm_power_deposition = data_obj.get_peak_SAR_VOP(temp_shim)
                #
                sar_shimmed = np.einsum("d, dczyx, c -> zyx", temp_shim.conjugate(), Q_container, temp_shim)
                peak_SAR = np.max(sar_shimmed.real)
                #
                temp_dict['opt_shim'] = temp_shim
                temp_dict['peak_SAR'] = peak_SAR
                temp_dict['b1p_nrmse'] = nrmse
                result.append(temp_dict)
            #
            result_dict_list = hmisc.listdict2dictlist(result)
            result_dict_list['opt_shim'] = [[list(x.real), list(x.imag)] for x in result_dict_list['opt_shim']]
            json_ser_obj = json.dumps(result_dict_list)
            # Create the file path for the optimal thing..
            ddest_dict = os.path.join(json_dest_dir, f'opt_shim_{str(ii).zfill(2)}.json')
            with open(ddest_dict, 'w') as f:
                f.write(json_ser_obj)
