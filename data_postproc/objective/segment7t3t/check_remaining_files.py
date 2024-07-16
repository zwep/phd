import csv
import shutil
import os

"""
There are files which we have not processed yet.. which are those
"""

dcsv = '/data/cmr7t3t/cmr7t/overview_ed_es_slice.csv'
dlabel = '/data/cmr7t3t/cmr7t/Label_ED_ES'
dimg = '/data/cmr7t3t/cmr7t/Image_ED_ES'

current_label_files = os.listdir(dlabel)
current_img_files = os.listdir(dimg)

missing_label_file = []
missing_img_file = []
with open(dcsv, 'r') as f:
    csv_obj = csv.reader(f)
    header = next(csv_obj)
    for i_row in csv_obj:
        row_dict = dict(zip(header, i_row))
        cur_name = row_dict['Subject name']
        cur_owner = row_dict['Owner']
        remove_ind = row_dict['remove']
        if remove_ind != '1':
            if 'ED_' + cur_name not in current_label_files:
                missing_label_file.append((cur_name, cur_owner))
            if 'ED_' + cur_name not in current_img_files:
                missing_img_file.append((cur_name, cur_owner))

print("Missing label files")
for i in missing_label_file:
    ifile = i[0]
    print(f'{ifile}', ' ' * (30 - len(ifile)), f'- Owner: {i[1]}')

print("Missing image files")
for i in missing_img_file:
    ifile = i[0]
    print(f'{ifile}', ' ' * (30 - len(ifile)), f'- Owner: {i[1]}')
