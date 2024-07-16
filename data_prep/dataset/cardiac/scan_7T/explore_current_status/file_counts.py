import helper.misc as hmisc
import collections
import re
import os
import data_prep.dataset.cardiac.scan_7T.helper_scan_7T as helper_scan_7T
"""
We have a lot of images...
But how many?
.. Of each type?
.. Of each Volunteer?
.. Have left after unfolding?

"""


# here we explore the file counts in the scan folder
scan_folder = '/media/bugger/MyBook/data/7T_scan/cardiac'
unfolded_folder = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac'
unfolded_cardiac_cine = '/media/bugger/MyBook/data/7T_data/cartesian_cardiac_cine'
unfolded_retro_cine = '/media/bugger/MyBook/data/7T_data/radial_retro_cardiac_cine'
unfolded_triggered_cine = '/media/bugger/MyBook/data/7T_data/radial_trigger_cardiac_cine'


all_scan_files, all_scan_directories = helper_scan_7T.get_date_files(scan_folder)
all_scan_files = list(set(all_scan_files))

v_number_directories = [x for x in all_scan_directories if os.path.basename(x).startswith('V')]
print('Number of V numbers', len(v_number_directories))

# This shows that I have 1298 unique files...
unique_scan_file_dict = helper_scan_7T.get_unique_date_time_files(all_scan_files)
print('Number of unique dates', len(unique_scan_file_dict))


helper_scan_7T.print_file_occurence(unique_scan_file_dict)


radial_2ch_list = []
for k, v in unique_scan_file_dict.items():
    if 'sa' in v[0] and 'radial' in v[0]:
        radial_2ch_list.append(k)

len(radial_2ch_list)

"""
But how many files are there in the unfolded directory...?
"""

v_number_list = os.listdir(unfolded_folder)
all_unfolded_files, all_unfolded_directories = helper_scan_7T.get_date_files(unfolded_folder)
unique_unfolded_file_dict = helper_scan_7T.get_unique_date_time_files(all_unfolded_files)
print('Number of unique dates', len(unique_unfolded_file_dict))
helper_scan_7T.print_file_occurence(unique_unfolded_file_dict)
# Need to remove the base name because of the folder structure
all_unfolded_directories = list(set([os.path.dirname(x) for x in all_unfolded_directories]))
print('Number of V numbers ', len(all_unfolded_directories))

# Okay so there are considerably less files in than we scanned. Good  to know
# Which V-numbers are we missing?


"""
How much are there in the cartesian scan things..?
"""

all_cart_files, _ = helper_scan_7T.get_date_files(unfolded_cardiac_cine)
unique_cart_file_dict = helper_scan_7T.get_unique_date_time_files(all_cart_files)
print('Number of unique dates', len(unique_cart_file_dict))
helper_scan_7T.print_file_occurence(unique_cart_file_dict)

all_retro_files, _ = helper_scan_7T.get_date_files(unfolded_retro_cine)
unique_retro_file_dict = helper_scan_7T.get_unique_date_time_files(all_retro_files)
print('Number of unique dates', len(unique_retro_file_dict))
all_trigger_files, _ = helper_scan_7T.get_date_files(unfolded_triggered_cine)
unique_trigger_file_dict = helper_scan_7T.get_unique_date_time_files(all_trigger_files)
print('Number of unique dates', len(unique_trigger_file_dict))

# What do I get when I combine the cartesian files and unfolded files. What is the overlap?
comb_all_files = all_cart_files + all_unfolded_files + all_retro_files + all_trigger_files
unique_comb_file_dict = helper_scan_7T.get_unique_date_time_files(comb_all_files)
print('Number of unique date time', len(unique_comb_file_dict))
helper_scan_7T.print_file_occurence(unique_comb_file_dict)

# I am still missing some files... which are they..?
# We want to compare these dicts..
set_comb_date_time = set(unique_comb_file_dict.keys())
set_scan_date_time = set(unique_scan_file_dict.keys())
print("Intersection", len(set_comb_date_time.intersection(set_scan_date_time)))
print("Difference comb - scan", len(set_comb_date_time.difference(set_scan_date_time)))
print("Difference scan - comb", len(set_scan_date_time.difference(set_comb_date_time)))
difference_list = list(set_scan_date_time.difference(set_comb_date_time))
# Find an example of p2ch/sa/4ch etc.
difference_dict = {k: v for k, v in unique_scan_file_dict.items() if k in difference_list}
helper_scan_7T.print_file_occurence(difference_dict)

difference_files = []
for itime_str in difference_list:
    if 'p2chV4' in ''.join(unique_scan_file_dict[itime_str]):
        difference_files.append(itime_str)
        # print(' Difference files ', itime_str)

len(difference_files)
sorted(difference_files, key=lambda x: int(x[4:8]))
# THERE ARE IMPORTANT PARTS MISSING STILL
# Now I want to check something..

# Ik moet sowieso nog 16x mensen/scans verwerken.
"""
"""