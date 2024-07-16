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

def _get_hourminute(x):
    # Returns the hour and minute of a file name with a specific string..
    return datetime.datetime.strptime(x[12:12 + 4], "%H%M")


def _get_time_difference(x, y):
    # Return difference in seconds
    return np.abs(_get_hourminute(x) - _get_hourminute(y)).seconds


"""
We have some undersampled 'testing data'

Some of these are also acquired in a fully sampled setting
We need to find those and evaluate their performance compared to what the models do


--> No idea how I populated `radial_trigger_cardiac_cine`


"""

d7tdata = '/media/bugger/MyBook/data/7T_data'
d7tscan = '/media/bugger/MyBook/data/7T_scan/cardiac'
ddest = '/media/bugger/MyBook/data/7T_data/cardiac_radial_us_fs'
ddest_csv_us_fs = os.path.join(ddest, 'us_fs_collection.csv')

# Create the directory...
hmisc.create_datagen_dir(ddest, type_list=['input', 'target', 'sensitivity'], data_list=[])

"""
Look in the scan data. Find in a directory containing .cpx data all the files that have 'high_time' in their name
and also files that have 'radial' in them and not 'high_time'.
If both are larger than 0, find the corresponding V-number and store that, the directory and all files in a list.

--> This will be used to match undersampled and fullysampled data 
--> high_time files are acquired WITH undersampling
--> files with only 'radial' in them are acquired with FULL sampling.
"""

radial_sampled_duo = []
for d, _, f in os.walk(d7tscan):
    f = [x for x in f if x.endswith('cpx')]
    if len(f):
        undersampled_files = [x for x in f if 'high_time' in x]
        fullysampled_files = [x for x in f if ('radial' in x) and (not 'high_time' in x)]
        if len(undersampled_files) and len(fullysampled_files):
            result_re = re_v_number.findall(d)
            if len(result_re):
                v_number = result_re[0]
            else:
                v_number = None
            radial_sampled_duo.append((v_number, d, undersampled_files, fullysampled_files))


# We found the existence of V-numbers that have high_time and fully sampled files
default_dict = {'vnumber': None, 'anatomy': None, 'filename': None}
data_collection = []
not_found_files = []
for i_container in radial_sampled_duo:
    v_number, d, undersampled_files, fullysampled_files = i_container
    # For each 'undersampled_files' file, find the closest (in time) fully sampled file.
    # For these two files, we search for a stored .h5 or .npy file in the 7T data directory.
    for i_file_undersampled_files in undersampled_files:
        fullysampled_files_file = min(fullysampled_files, key=lambda x: _get_time_difference(i_file_undersampled_files, x))
        i_anatomy = '_'.join([x for x in ANATOMY_LIST + ['trans'] if x in i_file_undersampled_files])
        temp_dict = {'vnumber': v_number, 'anatomy': i_anatomy, 'filename_undersampled_files': os.path.join(d, i_file_undersampled_files),
                     'filename_fullysampled_files': os.path.join(d, fullysampled_files_file)}
        data_collection.append(temp_dict)

temp_pandas = pd.DataFrame.from_dict(data_collection)
temp_pandas.to_csv(ddest_csv_us_fs, index=False)

