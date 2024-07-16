import os
"""
We got some patient data on a shared network folder on a windows computer...

This script can be run remotly to check the number of patients (or T1w scans) we have gathered thus far
(Remotely means on your own personal PC, connect via Remnia)

(last update 9 feb)
Total number of patients (remotely): 97
Total number of patients (local): 31
"""

data_path = "\\\\ds\data\BEELD\Wetenschap\AI_for_raw_data"
data_path = "/media/bugger/MyBook/data/3T_scan/head"
dir_found = [os.path.join(data_path, x) for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x))]
filter_ext = 'lab'

print("Date: ", " " * (15 - len("Date: ")), "#T1w files")
total_n_files = 0
for isubdir in sorted(dir_found):
    date_name = os.path.basename(isubdir)
    file_list = os.listdir(isubdir)
    # filter_file_list = [x for x in file_list if x.endswith(filter_ext)]
    scanner_files = [x for x in file_list if 'T1W_3D_TFE.raw' in x]
    n_files = len(scanner_files)
    first_column = f"\t {date_name}"
    second_column = f"\t {n_files}"
    total_n_files += n_files
    print(first_column, " " * (20 - len(first_column) + len(str(n_files)) - 1), second_column)

print(f"\n Total number of files {total_n_files}")