import os
import numpy as np
import helper.misc as hmisc
import nibabel
from objective_configuration.segment7T3T import get_path_dict
"""

"""

dir_dict = get_path_dict('mm2')
dimg = dir_dict['dimg']
file_list = os.listdir(dimg)
res = []
for i_file in file_list:
    print(i_file)
    file_path = os.path.join(dimg, i_file)
    nib_obj = nibabel.load(file_path)
    zooms = nib_obj._header.get_zooms()
    print(zooms)
    res.append(zooms)

np.mean(res, axis=0)