
import json
import os
import helper.misc as hmisc
import pandas as pd
import scipy.io
import numpy as np
"""
Here we create an overview of the selected mat-files in a .csv format table

This table is also used to create PNGs from the mat files
And this table is used in the further post processing pipeline
"""

ddata_new = '/data/cmr7t3t/cmr7t/RawData_newbatch/data_mat'
ddata_new_json = '/data/cmr7t3t/cmr7t/json_data_new_batch.json'

new_file_names = [hmisc.get_base_name(x) for x in os.listdir(ddata_new)]

# Here are the old file names, linked with their renamed file name
# E.g. from "v9_23012021_1243567_12_2_saV4.nrrd" to "subject_0004_0"
# We are going to use these file names to filter out any in the new batch
ddata_old_json = '/data/cmr7t3t/cmr7t/json_data.json'
with open(ddata_old_json, 'r') as f:
    file_name_conversion = json.loads(f.read())

old_file_names = [hmisc.get_base_name(x) for x in file_name_conversion['old_name']]
filtered_new_file_names = [False if x in old_file_names else True for x in new_file_names]
filtered_subject_names = [f"subject_1{str(file_id).zfill(3)}" if filtered_new_file_names[file_id] else "" for file_id in range(len(filtered_new_file_names))]

pandas_dict = {"mat files": new_file_names, "not in previous": filtered_new_file_names, "subject name": filtered_subject_names}
A = pd.DataFrame.from_dict(pandas_dict)

# Now add the number of locations...

name_location_dict = {'mat files': [], 'location': []}
for i_file in os.listdir(ddata_new):
    file_mat = os.path.join(ddata_new, i_file)
    mat_obj = scipy.io.loadmat(file_mat)
    data_obj = np.squeeze(mat_obj['data'])
    nx, ny, ncard, nloc = data_obj.shape
    for i_loc in range(nloc):
        base_name = hmisc.get_base_name(i_file)
        name_location_dict['mat files'].append(base_name)
        name_location_dict['location'].append(i_loc)

B = pd.DataFrame.from_dict(name_location_dict)
full_table = pd.merge(A, B, on='mat files', how='inner')
full_table['visual ok'] = False
full_table['ED slice'] = None
full_table['ES slice'] = None
full_table = full_table.sort_values('mat files')
full_table = full_table.reset_index(drop=True)
full_table.to_csv('/data/cmr7t3t/cmr7t/overview_new_batch.csv', index=False)
