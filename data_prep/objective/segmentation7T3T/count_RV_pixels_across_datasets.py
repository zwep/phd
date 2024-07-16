import os
import helper.misc as hmisc
from objective_configuration.segment7T3T import get_path_dict
import numpy as np
import nibabel

"""
Maybe the ACDC class is so bad because it has little RV examples?

No. Not the case.
"""

training_sets = ['mm1a', 'mm1b', 'acdc']
RV_count_dict = {}
for i_data in training_sets:
    RV_count_dict.setdefault(i_data, [])
    ddict = get_path_dict(i_data)
    dimg = ddict['dimg']
    dlabel = ddict['dlabel']
    file_list = os.listdir(dlabel)
    for i_file in file_list:
        file_path = os.path.join(dlabel, i_file)
        A = hmisc.load_array(file_path)
        n_slice = A.shape[-1]
        A = A[:, :, n_slice//2]
        nib_obj = nibabel.load(file_path)
        zooms = nib_obj._header.get_zooms()
        pixel_spacing = zooms[0]
        RV_count = np.sum(A == 1)
        RV_opp = RV_count * pixel_spacing ** 2
        RV_count_dict[i_data].append(RV_count)

hmisc.print_dict(RV_count_dict)
hmisc.print_dict_mean_value(RV_count_dict)