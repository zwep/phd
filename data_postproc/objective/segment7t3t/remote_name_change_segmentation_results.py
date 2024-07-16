
"""
Here another script to change names
"""

# Some scribbles..
import os
import re
base_ddata = '/data/cmr7t3t/mms1/all_phases_mid/Vendor_A/Results/'
ddata_list = ['202210_model_CUT_Only_150',
              '202210_model_1_5T_CycleGAN',
              '202210_model_1_5T_CUT_CycleGAN',
              '202210_model_1_5T_CUT']

for ddata in ddata_list:
    path_ddata = os.path.join(base_ddata, ddata)
    for i_file in os.listdir(path_ddata):
        new_i_file = re.sub('_0000', '', i_file)
        source_file = os.path.join(path_ddata, i_file)
        target_file = os.path.join(path_ddata, new_i_file)
        os.rename(source_file, target_file)