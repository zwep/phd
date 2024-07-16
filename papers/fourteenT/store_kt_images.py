import pandas as pd
import numpy as np
import objective_helper.fourteenT as helper_14T
from objective_configuration.fourteenT import COIL_NAME_ORDER, \
    DDATA_KT_BETA_VOP, DPLOT_KT_BETA_VOP,\
    DPLOT_KT_BETA_POWER, DDATA_KT_BETA_POWER,\
    DDATA_KT_POWER, DPLOT_KT_POWER, \
    TARGET_FLIP_ANGLE, WEIRD_RF_FACTOR, OPTIMAL_KT_POWER, \
    DDATA_KT_VOP, DPLOT_KT_VOP

"""
Here we want to store the kT images we calculated in npy format
This will eventually make it easier to calculate metrics on these things..
"""



"""
DDATA_KT_POWER contains the results obtained by forward power regularization..

Here we 
- report the avg rf norm and head sar per coil
- store the flip angle map and SAR distr
"""


def store_coil_spoke(sel_coil, sel_spoke, ddata, dplot):
    _ = head_sar_dict.setdefault(sel_coil, {})
    visual_obj = helper_14T.StoreKtImage(ddata, dplot, sel_coil,
                                         weird_rf_factor=WEIRD_RF_FACTOR,
                                         flip_angle_factor=TARGET_FLIP_ANGLE)

    for file_name in visual_obj.output_design_files:
        norm_avg_rf_waveform = np.sum(np.mean(np.abs(visual_obj.get_unique_pulse_settings(file_name)), axis=1) ** 2, axis=0)
        print("Coil: ", sel_coil, "// Spoke", visual_obj._get_kt_num(file_name), "// Avg RF norm", norm_avg_rf_waveform)
#
    file_name = visual_obj._get_kt_file(sel_spoke)
    visual_obj.store_flip_angle(file_name)
    avg_power_deposition = visual_obj.store_time_avg_SAR(file_name)
#
    norm_avg_rf_waveform = np.sum(np.mean(np.abs(visual_obj.get_unique_pulse_settings(file_name)), axis=1) ** 2, axis=0)
    head_sar_dict[sel_coil]['rf_norm'] = norm_avg_rf_waveform
    head_sar_dict[sel_coil]['head_sar'] = avg_power_deposition / sel_spoke



sel_kt_spoke = 5
head_sar_dict = {}
for icoil, sel_coil in enumerate(COIL_NAME_ORDER):
    print(sel_coil)
    store_coil_spoke(sel_coil=sel_coil, sel_spoke=sel_kt_spoke, ddata=DDATA_KT_POWER, dplot=DPLOT_KT_POWER)

pd_frame = pd.DataFrame(head_sar_dict)
print((pd_frame[COIL_NAME_ORDER]).round(2).to_csv(sep='\t'))

# Do the same for the VOPs
sel_kt_spoke = 5
head_sar_dict = {}
for icoil, sel_coil in enumerate(COIL_NAME_ORDER):
    print(sel_coil)
    store_coil_spoke(sel_coil=sel_coil, sel_spoke=sel_kt_spoke, ddata=DDATA_KT_VOP, dplot=DPLOT_KT_VOP)

pd_frame = pd.DataFrame(head_sar_dict)
print((pd_frame[COIL_NAME_ORDER]).round(2).to_csv(sep='\t'))

