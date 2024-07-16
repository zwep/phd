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
We have created a database of scan file names..

Now we are going to link the radial and cartesian files with eachother..
"""

ddest = '/home/bugger/Documents/paper/undersampled_recon'
# Read both cartesian and radial database
radial_database = pd.read_csv(os.path.join(ddest, 'scan_radial_files.csv'))
cartesian_database = pd.read_csv(os.path.join(ddest, 'scan_cartesian_files.csv'))
# We use these databases below to check what we have left... and thus how much work needs to be done..
unfolded_cine_database = pd.read_csv(os.path.join(ddest, 'unfolded_cine.csv'))

# Filter/prep useless rows
cartesian_database = helper_scan_7T.filter_none_v_number(cartesian_database)
radial_database = helper_scan_7T.filter_none_v_number(radial_database)

cartesian_database = helper_scan_7T.filter_on_ext(cartesian_database)
radial_database = helper_scan_7T.filter_on_ext(radial_database)

cartesian_database = helper_scan_7T.filter_on_dir(cartesian_database)
radial_database = helper_scan_7T.filter_on_dir(radial_database)

cartesian_database = helper_scan_7T.merge_row_duplicates(cartesian_database, 'file_name')
radial_database = helper_scan_7T.merge_row_duplicates(radial_database, 'file_name')

# Select rows with a specific string in the file name
retrospective_radial_database = helper_scan_7T.filter_on_file_name(radial_database, 'retrospective')
trig_radial_database = helper_scan_7T.filter_on_file_name(radial_database, 'triggered')
dyn_radial_database = helper_scan_7T.filter_on_file_name(radial_database, 'dyn')
high_time_radial_database = helper_scan_7T.filter_on_file_name(radial_database, 'high_time')
no_trig_radial_database = helper_scan_7T.filter_on_file_name(radial_database, '(no_trig|no_cardiac_trig|no_breathing_trig)')
fast_radial_database = helper_scan_7T.filter_on_file_name(radial_database, 'fast')

# Find all the fully sampled radial files here
radial_db_list = [trig_radial_database, dyn_radial_database,
                  high_time_radial_database, no_trig_radial_database,
                  retrospective_radial_database, fast_radial_database]
selected_indices = [list(i_db.index) for i_db in radial_db_list]
selected_indices = list(itertools.chain(*selected_indices))
remaining_radial_database = radial_database.loc[radial_database.index.difference(selected_indices)]

"""
Now connect each radial file with its cartesian counter part...
Based on V-number and scan type..
This can possibily be a many-to-many relationship

Store these for later use 
"""


cart_remaining_radial = helper_scan_7T.merge_dataframes(cartesian_database, remaining_radial_database)
cart_retro_radial = helper_scan_7T.merge_dataframes(cartesian_database, retrospective_radial_database)
cart_dyn_radial = helper_scan_7T.merge_dataframes(cartesian_database, dyn_radial_database)
cart_trig_radial = helper_scan_7T.merge_dataframes(cartesian_database, trig_radial_database)
cart_high_time_radial = helper_scan_7T.merge_dataframes(cartesian_database, high_time_radial_database)
cart_no_trig_radial = helper_scan_7T.merge_dataframes(cartesian_database, no_trig_radial_database)
cart_fast_radial_radial = helper_scan_7T.merge_dataframes(cartesian_database, fast_radial_database)

