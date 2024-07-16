import numpy as np
import pandas as pd
import os
import data_prep.dataset.cardiac.scan_7T.helper_scan_7T as helper_scan_7T

"""

Here we simply read the scan files. Save it, with some filtering, and store that cartesian and radial files.

"""

# here we explore the file counts in the scan folder
scan_folder = '/media/bugger/MyBook/data/7T_scan/cardiac'
ddest = '/home/bugger/Documents/paper/undersampled_recon'

dataframe_scan = helper_scan_7T.get_data_frame_scan_files(scan_folder)

# Here we already store the radial and cartesian file databases
radial_database = dataframe_scan[dataframe_scan['file_name'].apply(lambda x: 'radial' in x)]
radial_database.to_csv(os.path.join(ddest, 'scan_radial_files.csv'), index=False)

cartesian_database = dataframe_scan[dataframe_scan['file_name'].apply(lambda x: 'radial' not in x)]
cartesian_database.to_csv(os.path.join(ddest, 'scan_cartesian_files.csv'), index=False)

dataframe_scan = helper_scan_7T.merge_row_duplicates(dataframe_scan)
dataframe_scan['date_time'] = pd.to_datetime(dataframe_scan['date_time'])
dataframe_scan.to_csv(os.path.join(ddest, 'scan_cine_files.csv'), index=False)
