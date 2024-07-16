import numpy as np
import pandas as pd
import os
import data_prep.dataset.cardiac.scan_7T.helper_scan_7T as helper_scan_7T


ddest = '/home/bugger/Documents/paper/b1_shim'
scan_folder = '/media/bugger/MyBook/data/7T_scan/cardiac'
dataframe_scan = helper_scan_7T.get_data_frame_b1_shim_files(scan_folder)
# Make sure we really only have b1
id_sel_slice = dataframe_scan['slice_name'].apply(lambda x: x.lower() == 'b1shim')
dataframe_scan = dataframe_scan.loc[id_sel_slice]
# Make sure we get all the .cpx data
id_sel_ext = dataframe_scan['ext'].apply(lambda x: 'cpx' in x)
dataframe_scan = dataframe_scan.loc[id_sel_ext]
#
dataframe_scan = helper_scan_7T.merge_row_duplicates(dataframe_scan, 'file_name')
helper_scan_7T.resolve_concat_col(dataframe_scan, 'directory')

dataframe_scan.to_csv(os.path.join(ddest, 'b1_shim_series.csv'), index=False)
