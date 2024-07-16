import os
import numpy as np
import collections
import helper.misc as hmisc

"""
Needed to check if ALL the label files contianed valued in [0, 3]
"""

ddata = '/data/cmr7t3t/mms1/all_phases_mid/Vendor_A/Label'

for i_file in os.listdir(ddata):
    loaded_array = hmisc.load_array(os.path.join(ddata, i_file))
    if len(collections.Counter(loaded_array.ravel()).keys()) != 4:
        print(i_file, collections.Counter(loaded_array.ravel()))


"""
Loading array from SINA
"""
import os
import helper.array_transf as harray
import helper.misc as hmisc
ddata = '/data/cmr7t3t/results/ACDC_220121/seven2three_acdc_cut_NCE4_GAN2_np128_fe_211208/test_100/niftis/cmr3t2cmr7t'
sel_file = os.path.join(ddata, os.listdir(ddata)[0])
sel_array = hmisc.load_array(sel_file)
print("Shape sina array ", sel_array.shape)
