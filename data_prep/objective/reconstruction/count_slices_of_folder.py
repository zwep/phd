import helper.misc as hmisc
import pandas as pd
import re
import itertools
import os
from objective_configuration.reconstruction import DDATA

"""
I have made this before... but OK here we go

Path is relative to DDATA
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-path', help='Example: mixed/train/input')

p_args = parser.parse_args()
path = p_args.path
path = os.path.join(DDATA, path)

file_list = [x for x in os.listdir(path) if x.endswith('h5')]
total_slices = 0

# Extract Vunmber -- this only works for inference data..
dvnumber_csv = os.path.join(DDATA, 'vnumber_overview.csv')
vnumber_csv = pd.read_csv(dvnumber_csv)

re_v_number = re.compile('(V[0-9]_[0-9]*)')
v_numbers = set(itertools.chain(*[re_v_number.findall(x) for x in file_list]))

# Extract MM1A patient
re_mm1a = re.compile('to_(.*)_loc')
mm1a_numbers = set(itertools.chain(*[re_mm1a.findall(x) for x in file_list]))

v_number_list = []
for i_file in file_list:
    base_name = hmisc.get_base_name(i_file)
    if '_sa_' in base_name:
        base_name = base_name[:-3]
    print(base_name)
    sel_file = os.path.join(path, i_file)
    temp_array = hmisc.load_array(sel_file, data_key='kspace')
    n_slice = temp_array.shape[0]
    total_slices += n_slice
    if base_name in vnumber_csv['filename'].tolist():
        vnumber_sel = vnumber_csv.loc[vnumber_csv['filename'] == base_name]['vnumber'].values
        v_number_list.extend(vnumber_sel)

print(f'--- Directory {path} -----')
print('Total amount of files ', len(file_list))
print('Total amount of slices ', total_slices)
print('Number of unique Vnumbers ', len(v_numbers))
print('Number of unique Vnumbers ', len(set(v_number_list)))
print('Number of unique MM1a ', len(mm1a_numbers))