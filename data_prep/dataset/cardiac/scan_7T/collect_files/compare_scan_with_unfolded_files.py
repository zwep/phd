import numpy as np
import pandas as pd
import os
import data_prep.dataset.cardiac.scan_7T.helper_scan_7T as helper_scan_7T
from data_prep.dataset.cardiac.scan_7T.visualize_unfolded_files import FileGatherer

"""

There are files in the scan folder.. and in other folders which are unfolded

Here we gather all the files from the scan folder and put them in a .csv

The scan files are also used to count how much unfolding we have done


"""

ddest = '/home/bugger/Documents/paper/undersampled_recon'
ddest_csv = os.path.join(ddest, 'to_be_processed.csv')
ddest_data = '/media/bugger/MyBook/data/7T_data/cardiac_cine_mat'

# These are the ones we wanted
file_data_base = pd.read_csv(ddest_csv)
planned_files = file_data_base['file_name'].values

# These are the ones we have done
file_gather_obj = FileGatherer.FileGather(ddest_data)
executed_files = file_gather_obj.file_list_ext
import helper.misc as hmisc
executed_files = [hmisc.get_base_name(x) for x in executed_files]

# This is the match between them
n_files_exec = len(executed_files)
n_files_planned = len(planned_files)
print('Executed files ', n_files_exec)
print('Planned files ', n_files_planned)

difference_executed_planned = set(executed_files).difference(planned_files)
difference_planned_executed = set(planned_files).difference(executed_files)
union_planned_executed = set(executed_files).union(planned_files)

print('Union files ', len(union_planned_executed))
print('Difference exec with planned files ', len(difference_executed_planned))
print('Difference planned with exec files ', len(difference_planned_executed))
