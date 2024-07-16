import itertools
import helper.misc as hmisc
import numpy as np
import collections
import re
import importlib
import pandas as pd
import datetime

import os
import data_prep.dataset.cardiac.scan_7T.helper_scan_7T as helper_scan_7T
import matplotlib.pyplot as plt

"""
I think I need to do this...
Push them all through the same pipeline. Because the .par/.cpx files already have that black cirlce around the radial files...

"""

ddest = '/home/bugger/Documents/paper/undersampled_recon'
ddest_data = '/media/bugger/MyBook/data/7T_data/cardiac_cine_mat'

# Read both cartesian and radial database
radial_database = pd.read_csv(os.path.join(ddest, 'scan_radial_files.csv'))
cartesian_database = pd.read_csv(os.path.join(ddest, 'scan_cartesian_files.csv'))

# Also read the files that we currently have processed
dataframe_scan = helper_scan_7T.get_data_frame_scan_files(ddest_data)

# Filter/prep useless rows
cartesian_database = helper_scan_7T.filter_none_v_number(cartesian_database)
radial_database = helper_scan_7T.filter_none_v_number(radial_database)

cartesian_database = helper_scan_7T.filter_on_ext(cartesian_database)
radial_database = helper_scan_7T.filter_on_ext(radial_database)

cartesian_database = helper_scan_7T.filter_on_dir(cartesian_database)
radial_database = helper_scan_7T.filter_on_dir(radial_database)

# Merge rows based on the same file name
cartesian_database = helper_scan_7T.merge_row_duplicates(cartesian_database, 'file_name')
radial_database = helper_scan_7T.merge_row_duplicates(radial_database, 'file_name')

# Resolve concatted rows in the directory column
helper_scan_7T.resolve_concat_col(cartesian_database, 'directory')
helper_scan_7T.resolve_concat_col(radial_database, 'directory')

# Obtain all the sense-refscan files
sense_file_list = []
for i, irow in cartesian_database.iterrows():
    file_list = sorted(os.listdir())
    sel_ext = '.lab'
    sel_dir = irow['directory']
    sel_file = irow['file_name']
    sense_file = helper_scan_7T.get_sense_file(x_dir=sel_dir, x_name=sel_file)
    sense_file_list.append(sense_file)

cartesian_database['sense_file'] = sense_file_list
radial_database['sense_file'] = [None] * len(radial_database)

# Combine all the files and make sure that the destination folder exists
to_be_loaded = pd.concat([cartesian_database, radial_database])
to_be_loaded['ddest_file'] = None
for i, irow in to_be_loaded.iterrows():
    i_ddest = os.path.join(ddest_data, irow['v_number'], irow['slice_name'])
    irow['ddest_file'] = os.path.join(i_ddest, irow['file_name'])
    if not os.path.isdir(i_ddest):
        os.makedirs(i_ddest)

# Filter out the already known files
to_be_loaded_filtered = pd.merge(to_be_loaded, dataframe_scan[['v_number', 'file_name', 'slice_name']], on=('v_number', 'file_name', 'slice_name'), indicator=True, how='outer')
to_be_loaded_filtered = to_be_loaded_filtered[to_be_loaded_filtered['_merge'] == 'left_only'].reset_index(drop=True)
del to_be_loaded_filtered['_merge']

# Write the data to two locations
to_be_loaded_filtered.to_csv(os.path.join(ddest, 'to_be_processed.csv'), index=False)
to_be_loaded_filtered.to_csv(os.path.join(ddest_data, 'to_be_processed.csv'), index=False)


# On Windows...
import sys
sys.path.append(r'F:\code_backup\pytorch_in_mri')
import pandas as pd
import os
ddest_data = 'F:/data/7T_data/cardiac_cine_mat'
to_be_loaded = pd.read_csv(r'F:\data\7T_data\cardiac_cine_mat\to_be_processed.csv')
dataframe_scan = helper_scan_7T.get_data_frame_scan_files(ddest_data)

to_be_loaded_filtered = pd.merge(to_be_loaded, dataframe_scan[['v_number', 'file_name', 'slice_name']], on=('v_number', 'file_name', 'slice_name'), indicator=True, how='outer')
to_be_loaded_filtered['_merge'].value_counts()
to_be_loaded_filtered = to_be_loaded_filtered[to_be_loaded_filtered['_merge'] == 'left_only'].reset_index(drop=True)

del to_be_loaded_filtered['_merge']

to_be_loaded_filtered.to_csv(r'F:\data\7T_data\cardiac_cine_mat\to_be_processed_nov.csv', index=False)