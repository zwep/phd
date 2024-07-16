"""
This involved the splitting of a bipartite graph..

Was used to make train/test totally independent
"""

import re
import collections
import numpy as np
import helper.misc as hmisc

# Result_files contained the combination of two sets of files
result_files = []


# Here we get the IDs that we find in the results
# Part are from cardiac arrays.. part are scan arrays
# We want to find a division such that we have independent sets..
re_cardiac = re.compile('(Cardiac.*Data)')
re_scan = re.compile('([0-9]{7})')

cardiac_solutions = [re_cardiac.findall(x)[0] for x in result_files if re_cardiac.findall(x)]
cardiac_counts = collections.Counter(cardiac_solutions)

cardiac_list_counts = list(cardiac_counts.values())
cardiac_list_keys = np.array(list(cardiac_counts.keys()))
n_total = sum(cardiac_list_counts)
n_train = int(0.7 * n_total)
bool_cardiac_solution = hmisc.get_subset_sum(cardiac_list_counts, n_train)

sel_cardiac_train = cardiac_list_keys[bool_cardiac_solution]
sel_cardiac_test = cardiac_list_keys[~bool_cardiac_solution]

# Get the train/test split based on the cardiac split based on the subset sum
sel_solutions_train = [x for x in result_files if re_cardiac.findall(x)[0] in sel_cardiac_train]
sel_solutions_test = [x for x in result_files if re_cardiac.findall(x)[0] in sel_cardiac_test]
print('Percentage between train/test ', len(sel_solutions_test) / (len(sel_solutions_train) + len(sel_solutions_test)))

# Get the scan-ids with the current division
sel_scan_solutions_train = [re_scan.findall(x)[0] for x in sel_solutions_train if re_scan.findall(x)]
sel_scan_solutions_test = [re_scan.findall(x)[0] for x in sel_solutions_test if re_scan.findall(x)]

# Check which scan ids are both in test AND train (we do not want that)
intersection_scan = list(set(sel_scan_solutions_train).intersection(set(sel_scan_solutions_test)))
# Keep those that are not intersecting with others...
sel_scan_solutions_test = list(set(sel_scan_solutions_test).difference(set(intersection_scan)))
sel_scan_solutions_train = list(set(sel_scan_solutions_train).difference(set(intersection_scan)))

# Filter the sel scan solutions based on the filtering from above
sel_solutions_test = [x for x in sel_solutions_test if re_scan.findall(x)[0] in sel_scan_solutions_test]
sel_solutions_train = [x for x in sel_solutions_train if re_scan.findall(x)[0] in sel_scan_solutions_train]
print('Percentage between train/test ', len(sel_solutions_test) / (len(sel_solutions_train) + len(sel_solutions_test)))

n_test = len(sel_solutions_test)
n_train = len(sel_solutions_train)

# Double check on independence between the sets...
temp_set_1 = set([re_cardiac.findall(x)[0] for x in sel_solutions_train])
temp_set_2 = set([re_cardiac.findall(x)[0] for x in sel_solutions_test])
print('Amount of overlap in Cardiac ID', len(temp_set_1.intersection(temp_set_2)))

temp_set_1 = set([re_scan.findall(x)[0] for x in sel_solutions_train])
temp_set_2 = set([re_scan.findall(x)[0] for x in sel_solutions_test])
print('Amount of overlap in Scan ID', len(temp_set_1.intersection(temp_set_2)))
