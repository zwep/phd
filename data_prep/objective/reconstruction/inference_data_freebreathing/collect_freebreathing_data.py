import shutil

import helper.misc as hmisc
import re
import helper.plot_class as hplotc
import itertools
from objective_helper.reconstruction import scan2direct_array, resize_array, mat2direct
import numpy as np
import os
import h5py


"""
Test
"""

dscan = '/media/bugger/MyBook/data/7T_scan/cardiac'
ddest = '/media/bugger/MyBook/data/7T_data/freebreathing_raw'
if not os.path.isdir(ddest):
    os.makedirs(ddest)
re_v_number = re.compile('(V[0-9]_[0-9]*)')

retro_mat = hmisc.find_all_files_in_dir('free_breathing', dscan, ext='raw')
for i_file in retro_mat:
    base_name = hmisc.get_base_name(i_file)
    datetime = base_name[:19]
    for i_ext in ['lab', 'raw', 'sin']:
        source_ext_file = hmisc.find_file_in_dir(datetime, dscan, ext=i_ext)
        dest_ext_file = os.path.join(ddest, base_name + f'.{i_ext}')
        shutil.copy((source_ext_file, dest_ext_file))