import helper.misc as hmisc
import re
import helper.plot_class as hplotc
import itertools

"""
Previously we have used triggered data (which was u.s.) and its fully sampled counterpart

Buuuut we also have retrospective triggered data
"""

ddata = '/media/bugger/MyBook/data/7T_data/'
dscan = '/media/bugger/MyBook/data/7T_scan/cardiac'
re_v_number = re.compile('(V[0-9]_[0-9]*)')

retro_npy = hmisc.find_all_files_in_dir('retrospective', ddata, ext='npy')
retro_h5 = hmisc.find_all_files_in_dir('retrospective', ddata, ext='h5')
retro_mat = hmisc.find_all_files_in_dir('retrospective', ddata, ext='mat')

# We have found a lot of retrospective collected files in .mat format
# This also contains complex data, and all the coil information. Which is nice.

# Now check which v-numbres belong to these
vnumber_dict = {}
for i_mat in retro_mat:
    date_time = hmisc.get_base_name(i_mat)[:19]
    found_files = hmisc.find_all_files_in_dir(date_time, dscan, ext='sin')
    if len(found_files):
        v_numbers = list(itertools.chain(*[re_v_number.findall(x) for x in found_files]))
        for i_vnum in v_numbers:
            vnumber_dict.setdefault(i_vnum, [])
            vnumber_dict[i_vnum].append(i_mat)

#$ I copied these V-numbers from the remote server
vnum_train_remote = ['V9_17068', 'V9_16654', 'V9_19528', 'V9_16830', 'V9_16655', 'V9_16656', 'V9_17911', 'V9_17913', 'V9_17914', 'V9_19527', 'V9_16834', 'V9_17994', 'V9_16935', 'V9_19529', 'V9_19530', 'V9_19526', 'V9_17069']
vnum_val_remote = ['V9_17912', 'V9_16829', 'V9_17915', 'V9_13975', 'V9_15934', 'V9_16051', 'V9_16657', 'V9_13518', 'V9_13296']
vnum_test_remote = ['V9_16936', 'V9_19531', 'V9_17067']
vnum_remote = set(vnum_test_remote).union(set(vnum_train_remote)).union(set(vnum_val_remote))
# The set below is empty. Which shows that only NEW anatomies are contained in this..
set(list(vnumber_dict.keys())).intersection(vnum_remote)