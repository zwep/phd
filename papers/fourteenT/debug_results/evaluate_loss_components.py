from objective_helper.fourteenT import VisualizeAllMetrics, DataCollector
import matplotlib.pyplot as plt
import scipy.io
import helper.misc as hmisc
import os
import objective_helper.fourteenT as helper_14T
from objective_configuration.fourteenT import COIL_NAME_ORDER, COLOR_DICT, \
    DPLOT_1KT_BETA_POWER, DDATA_1KT_BETA_POWER, \
    DPLOT_1KT_BETA_VOP, DDATA_1KT_BETA_VOP, \
    DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP, \
    DDATA_KT_BETA_POWER, DPLOT_KT_BETA_POWER, \
    COIL_NAME_ORDER_TRANSLATOR, RF_SCALING_FACTOR_1KT, WEIRD_RF_FACTOR, TARGET_FLIP_ANGLE, SUBDIR_RANDOM_SHIM
from objective_configuration.fourteenT import CALC_OPTIONS, DDATA, MAX_ITER, TARGET_B1
import os
import numpy as np
from objective_helper.fourteenT import ReadMatData, OptimizeData

"""
I want to see the loss components of each optimization piece...
"""

icoil = 0
selected_coil = COIL_NAME_ORDER[icoil]
coil_plot_name = COIL_NAME_ORDER_TRANSLATOR[selected_coil]
dpower = '/data/seb/paper/14T/plot_body_thomas_mask_rmse_power'
dsar = '/data/seb/paper/14T/plot_body_thomas_mask_rmse_sar'

file_list = os.listdir(DDATA)
mat_files = [x for x in file_list if x.endswith('mat')]
sel_mat_file = [x for x in mat_files if selected_coil in x][0]

mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)

"""
Analyze loss components of the regularization on Power
"""

visual_obj_power = VisualizeAllMetrics(ddest=dpower, opt_shim_str=f'opt_shim_00')
optim_json = visual_obj_power.optimized_json_data[selected_coil]
data_obj_power = OptimizeData(ddest=dpower, objective_str='rmse_power', mat_reader=mat_reader, full_mask=True, type_mask='thomas_mask')
for ii, (inrmse, inorm) in enumerate(zip(optim_json['residual'], optim_json['norm_power'])):
    print(np.round(inrmse, 2), np.round(data_obj_power.lambda_range[ii] * inorm, 2))


"""
Similar, but for SAR based regularization"""
visual_obj_sar = VisualizeAllMetrics(ddest=dsar, opt_shim_str=f'opt_shim_00')
optim_json_sar = visual_obj_sar.optimized_json_data[selected_coil]

data_obj_sar = OptimizeData(ddest=dsar, objective_str='rmse_sar', mat_reader=mat_reader, full_mask=True, type_mask='thomas_mask')
for ii, (inrmse, isar) in enumerate(zip(optim_json_sar['residual'], optim_json_sar['peak_SAR'])):
    print(np.round(inrmse, 2), np.round(data_obj_sar.lambda_range[ii] * isar, 2))


"""
What about a random shim...?
"""
drandom = '/data/seb/paper/14T/plot_body_thomas_mask_rmse_power'
cpx_key = 'random_shim'
visual_obj = VisualizeAllMetrics(ddest=drandom, opt_shim_str=f'opt_shim_00')

json_path = os.path.join(drandom, SUBDIR_RANDOM_SHIM, selected_coil, f'{cpx_key}.json')
if os.path.isfile(json_path):
    random_json = visual_obj._load_json(json_path, cpx_key=cpx_key)

for ii, (inrmse, inorm, isar) in enumerate(zip(random_json['residual'], random_json['norm_power'], random_json['peak_SAR'])):
    print(np.round(inrmse, 2), np.round(data_obj_sar.lambda_range[ii] * inorm, 2), np.round(data_obj_sar.lambda_range[ii] * isar, 2))
