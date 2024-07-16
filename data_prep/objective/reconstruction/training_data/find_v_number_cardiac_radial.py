import os
import datetime
import helper.misc as hmisc
import helper.plot_class as hplotc
import numpy as np
from objective_configuration.reconstruction import ANATOMY_LIST
import pandas as pd
import re
from objective_helper.reconstruction import convert_to_sos

re_v_number = re.compile('(V[0-9]_[0-9]*)')

d7tdata = '/media/bugger/MyBook/data/7T_data'
d7tscan = '/media/bugger/MyBook/data/7T_scan/cardiac'
dradial_base = '/media/bugger/MyBook/data/7T_data/cardiac_radial'


"""
We want to find each V-number of each .npy file  and store that in a csv
"""

# Needed to find the V-number
data_collection = []
# loop over all train, validation and anatomies and check which V-numbers belongs to that name.
default_dict = {'vnumber': None, 'anatomy': None, 'data_type': None, 'filename': None}
for i_anatomy in ANATOMY_LIST:
    # Sometimes I name it 2ch sometimes partly-2ch (p2ch)
    if i_anatomy == '2ch':
        i_anatomy = 'p' + i_anatomy
    dpath = os.path.join(dradial_base + f"_{i_anatomy}")
    file_list = os.listdir(dpath)
    filename_list = [hmisc.get_base_name(x) for x in file_list if x.endswith('npy')]
    for i_filename in filename_list:
        # The last characters of short axis files contain a location number (_01, _02, ..)
        if i_anatomy == 'sa':
            i_filename = i_filename[:-3]
        # Find .par files that contain the date + timestamp
        # Extract the V-numbers from those files
        found_dir = hmisc.find_file_in_dir(i_filename[:19], dir_name=d7tscan, ext='par')
        if found_dir:
            result_re = re_v_number.findall(found_dir)
            if len(result_re):
                temp_dict = {'vnumber': result_re[0], 'anatomy': i_anatomy,  'filename': i_filename}
                data_collection.append(temp_dict)
            else:
                data_collection.append(default_dict)
        else:
            data_collection.append(default_dict)

temp_pandas = pd.DataFrame.from_dict(data_collection)
temp_pandas.to_csv('/home/bugger/Documents/paper/reconstruction/vnumber_overview.csv', index=False)
