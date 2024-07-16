import helper.misc as hmisc
import collections
import re
import os
import pandas as pd
import datetime

"""
Helper functions for the scan-7T database
"""

import numpy as np


def get_sense_file(x_dir, x_name):
    re_date_time_obj = re.compile('_([0-9]{8}_[0-9]{7})_')
    x_date_time = re_date_time_obj.findall(x_name)[0]
    file_list = sorted([x for x in os.listdir(x_dir) if not x.endswith('log')])
    file_list_date_time = [re_date_time_obj.findall(x)[0] for x in file_list if re_date_time_obj.findall(x)]
    # sort the chronologically.. hopefully...
    arg_index = np.argsort(file_list_date_time)
    file_list = np.array(file_list)[arg_index]
    file_list_date_time = list(np.array(file_list_date_time)[arg_index])
    # print(file_list)
    index_in_files = file_list_date_time.index(x_date_time)
    prev_sense_files = [x for x in file_list[:index_in_files] if 'sense' in x]
    if len(prev_sense_files) == 0:
        print(x_dir, x_name)
        last_sense_file = 'none'
    else:
        last_sense_file = prev_sense_files[-1]
        last_sense_date_time = re_date_time_obj.findall(last_sense_file)[0]
        # Now we have the sense-files that all belong to the same date-time as the last one..
        # This is to prevent cases where we have multiple sense files.
        # Next step is to get the .lab extension file
        last_sense_file_list = [x for x in prev_sense_files if re_date_time_obj.findall(x)[0] == last_sense_date_time]
        ext_list = [hmisc.get_ext(x) for x in last_sense_file_list]
        if '.lab' in ext_list:
            ext_index = ext_list.index('.lab')
        else:
            ext_index = 0
        last_lab_sense_file = last_sense_file_list[ext_index]
        last_sense_file = hmisc.get_base_name(last_lab_sense_file)
    return last_sense_file


def resolve_concat_col(x_df, str_col):
    """
    Specific function to correct V-numbers with a comma in their name due to merging
    :param x_df:
    :return:
    """
    x_df[str_col].fillna('').astype("string").apply(lambda x: ', ' in x)
    ind_comb_v_number = x_df[str_col].apply(lambda x: ', ' in str(x))
    extracted_v_number = x_df.loc[ind_comb_v_number, str_col].apply(lambda x: sorted(x.split(','))[-1])
    print(f'Number of {str_col} with concatenation ', ind_comb_v_number.sum())
    x_df.loc[ind_comb_v_number, str_col] = extracted_v_number


def filter_none_v_number(x_df):
    """
    Filter on correct v-numbers
    Select only those that have a proper vlaue in them
    :param x_df:
    :return:
    """
    ind_none_v_number = x_df['v_number'].apply(lambda x: x == 'None')
    ind_nan_v_number = x_df['v_number'].apply(lambda x: pd.isna(x))
    ind_null_v_number = ind_none_v_number | ind_nan_v_number
    print('Removing None V-numbers ', ind_null_v_number.sum())
    x_df = x_df.loc[~ind_null_v_number]
    return x_df.copy(deep=True)


def delete_columns(x_df, filter):
    """
    Delete specific columns with a filter. This is not a regex-filter
    :param x_df:
    :param filter:
    :return:
    """
    x_col = x_df.columns
    for i_col in x_col:
        if filter in i_col:
            print('Deleting column ', i_col)
            del x_df[i_col]


def apply_filter_on_str(x, filter_str):
    """
    Can be used in combination with apply on pandas DFs
    If it finds the filterstr, it does return it
    :param x:
    :param filter_str:
    :return:
    """

    return_value = True
    re_obj = re.findall(filter_str, x)
    if len(re_obj) == 0:
        return_value = False
    return return_value


def filter_on_ext(x_df, suffix=''):
    """
    Explicit method to filter on specific extension we need
    :param x_df:
    :param suffix:
    :return:
    """

    ext_ind = x_df['ext' + suffix].apply(lambda x: apply_filter_on_str(x, '\.lab') &
                                                   apply_filter_on_str(x, '\.raw') &
                                                   apply_filter_on_str(x, '\.sin'))
    print('Removing extensions ', (~ext_ind).sum())
    x_df = x_df.loc[ext_ind]
    return x_df.copy(deep=True)


def filter_on_dir(x_df, suffix=''):
    """
    Explicit method for removing directories that contain `matdata`
    :param x_df:
    :param suffix:
    :return:
    """
    dir_ind = x_df['directory' + suffix].apply(lambda x: apply_filter_on_str(x, 'matdata'))
    print('Removing directories ', (dir_ind).sum())
    x_df = x_df.loc[~dir_ind]
    return x_df.copy(deep=True)


def filter_on_file_name(x_df, filter_str, suffix=''):
    """
    A bit more general method to select rows with a specific string in their name
    Also reports how many we found. Returns a copy
    :param x_df:
    :param filter_str:
    :param suffix:
    :return:
    """

    file_name_ind = x_df['file_name' + suffix].apply(lambda x: apply_filter_on_str(x, filter_str))
    print('Selecting number of files: ', (file_name_ind).sum())
    x_df = x_df.loc[file_name_ind]
    return x_df.copy(deep=True)


def merge_dataframes(x_df, y_df):
    """
    This is how we merge cartesian- and radial-files.
    We base it on v-number and slice name.
    :param x_df:
    :param y_df:
    :return:
    """

    sel_col = ['v_number', 'slice_name', 'file_name', 'directory', 'date_time']
    print('n x_df :', len(x_df), '\nn y_df: ', len(y_df))
    x_y_df = pd.merge(x_df[sel_col], y_df[sel_col], on=['v_number', 'slice_name'], suffixes=('_cart', '_radial'))
    print('Resulting merge: ', len(x_y_df))
    return x_y_df


def get_unique_date_time_files(file_list):
    """
    The files from the scanner have a unique identifyer
    This is a combination of the date and time (?). The time has 7 digits, whic is a bit odd though
    Given a list of files, we create a dict with the date_time's as keys
    and all associated files with that as value in a list
    :param file_list:
    :return:
    """

    re_date_time_obj = re.compile('_([0-9]{8}_[0-9]{7})_')
    unique_file_dict = {}
    for i_file in file_list:
        date_time = re_date_time_obj.findall(i_file)[0]
        _ = unique_file_dict.setdefault(date_time, [])
        unique_file_dict[date_time].append(i_file)
    return unique_file_dict


def get_date_files(ddir):
    """
    Given a directory, we return all the files and (unique) directories
    that have files in them with a date format
    :param ddir:
    :return:
    """

    re_date_obj = re.compile('_([0-9]{8})_')
    all_date_files = []
    all_directories = []
    for d, _, f in os.walk(ddir):
        filter_f = [x for x in f if re_date_obj.findall(x)]
        if len(filter_f):
            all_date_files.extend(filter_f)
            all_directories.append(d)
    return all_date_files, all_directories


def get_dir_date_files(ddir):
    """
    Here we return a list of tuples. Where each tuple has a directory
    and a list of files that all contain a date in the file name
    :param ddir: 
    :return: 
    """

    re_date_obj = re.compile('_([0-9]{8})_')
    all_date_files = []
    for d, _, f in os.walk(ddir):
        filter_f = [x for x in f if re_date_obj.findall(x)]
        if len(filter_f):
            all_date_files.append((d, filter_f))
    return all_date_files


def print_file_occurence(unique_file_dict, filter_str_list=None):
    """
    Given a dictionary with unique date_time as keys and files as values
    We display a count of the last word in each file name
    THis gives us an idea what kind of files we are dealing with
    :param unique_file_dict:
    :param filter_str_list:
    :return:
    """

    if filter_str_list is None:
        filter_str_list = ['survey', '2ch', '4ch', 'sa']
    # Content of these unique dates...
    list_of_last_names = []
    for k, v in unique_file_dict.items():
        # Simply select the first one.. Makes not matter
        last_name = hmisc.get_base_name(v[0]).split('_')[-1]
        list_of_last_names.append(last_name)

    for istr in filter_str_list:
        print('\n')
        list_of_slice_names = [x for x in list_of_last_names if istr in x]
        count_dict = collections.Counter(list_of_slice_names)
        for k, v in count_dict.items():
            print(k, v)


def get_data_frame_scan_files(ddir):
    """
    Given a directory, we create a dataframe of all the entries we find
    Each entry is characterized by
    The V-number, file name, date time, scan type (p2ch, 4ch, ..) and ext (.lab, .sin, ...)

    Here we also exclude specific files, shown below (i.e. survey, b1, b0, senserefscan)
    :param ddir:
    :return:
    """
    # The scane name is based on the regex shown below.
    dir_date_files = get_dir_date_files(ddir)

    # Create columns to check content in other folders.
    exclude_files = ['survey', 'sens', 'b1', 'b0', 'bssf', 'fair',
                     'steam', 'refscan', 'ffedyn', 't2w', 'timeslip',
                     'SCANPHYSLOG', 'log']
    re_scan_str = re.compile('(tra|4ch|p2ch|sa)')

    scan_file_dataframe = pd.DataFrame()
    for i_dir, file_list in dir_date_files:
        v_number_obj = re.findall('(V[0-9]_[0-9]*)', i_dir)
        if v_number_obj:
            v_number = v_number_obj[0]
        else:
            print('No V-number found\n', i_dir)
            v_number = None
        v_number_dict = {'v_number': v_number, 'file_name': [], 'directory': [],
                         'date_time': [], 'ext': [], 'slice_name': []}
        filter_f = [x for x in file_list if not re.findall('|'.join(exclude_files), x)]
        unique_date_time_files = get_unique_date_time_files(filter_f)
        for k, v in unique_date_time_files.items():
            re_slice_name = re_scan_str.findall(v[0])
            if re_slice_name:
                slice_name = re_slice_name[0]
            else:
                slice_name = 'None'
            v_number_dict['directory'].append(i_dir)
            v_number_dict['date_time'].append(datetime.datetime.strptime(k[:15], '%d%m%Y_%H%M%S'))
            ext_list = [hmisc.get_ext(x) for x in v]
            if '.lab' in ext_list:
                ext_index = ext_list.index('.lab')
            else:
                ext_index = 0
            v_number_dict['file_name'].append(hmisc.get_base_name(v[ext_index]))
            extension_string = ', '.join(sorted(ext_list))
            v_number_dict['ext'].append(extension_string)
            v_number_dict['slice_name'].append(slice_name)

        v_number_df = pd.DataFrame(v_number_dict)
        scan_file_dataframe = pd.concat([scan_file_dataframe, v_number_df])

    scan_file_dataframe = scan_file_dataframe.reset_index(drop=True)
    return scan_file_dataframe


def get_data_frame_b1_shim_files(ddir):
    """
    Given a directory, we create a dataframe of all the entries we find
    Each entry is characterized by
    The V-number, file name, date time, scan type (b1_shim) and ext (.lab, .sin, ...)

    -- This is a duplicate of the get_data_frame_scan_files.. atleast, almost
    I am lazy atm. Could combine these two.
    :param ddir:
    :return:
    """
    # The scane name is based on the regex shown below.
    dir_date_files = get_dir_date_files(ddir)

    # Create columns to check content in other folders.
    exclude_files = ['survey', 'sens', 'b0', 'bssf', 'fair', '2ch', '4ch', 'tra', 'sa',
                     'steam', 'refscan', 'ffedyn', 't2w', 'timeslip',
                     'SCANPHYSLOG', 'log']
    re_b1_str = re.compile('(b1shim)')

    scan_file_dataframe = pd.DataFrame()
    for i_dir, file_list in dir_date_files:
        v_number_obj = re.findall('(V[0-9]_[0-9]*)', i_dir)
        if v_number_obj:
            v_number = v_number_obj[0]
        else:
            print('No V-number found\n', i_dir)
            v_number = None
        v_number_dict = {'v_number': v_number, 'file_name': [], 'directory': [],
                         'date_time': [], 'ext': [], 'slice_name': []}
        filter_f = [x for x in file_list if not re.findall('|'.join(exclude_files), x)]
        unique_date_time_files = get_unique_date_time_files(filter_f)
        for k, v in unique_date_time_files.items():
            re_slice_name = re_b1_str.findall(v[0])
            if re_slice_name:
                slice_name = re_slice_name[0]
            else:
                slice_name = 'None'
            v_number_dict['directory'].append(i_dir)
            v_number_dict['date_time'].append(datetime.datetime.strptime(k[:15], '%d%m%Y_%H%M%S'))
            ext_list = [hmisc.get_ext(x) for x in v]
            if '.cpx' in ext_list:
                ext_index = ext_list.index('.cpx')
            else:
                ext_index = 0
            v_number_dict['file_name'].append(hmisc.get_base_name(v[ext_index]))
            extension_string = ', '.join(sorted(ext_list))
            v_number_dict['ext'].append(extension_string)
            v_number_dict['slice_name'].append(slice_name)

        v_number_df = pd.DataFrame(v_number_dict)
        scan_file_dataframe = pd.concat([scan_file_dataframe, v_number_df])

    scan_file_dataframe = scan_file_dataframe.reset_index(drop=True)
    return scan_file_dataframe


def merge_row_duplicates(sel_dataframe, sel_col='date_time'):
    """
    In dataframes you can have duplicate rows in some sense
    Here, given a data frame, find rows that have duplicate values
    according to the values in a single specific column (date_time)
    Given this set of unique rows, concat these, take the unique values
    and join them again to construct a new unique row.
    Remove the old ones, add the new one. Tada, done.
    :param sel_dataframe:
    :param sel_col:
    :return:
    """
    x_value_counts = sel_dataframe[sel_col].value_counts()
    duplicate_values = x_value_counts[x_value_counts > 1].index
    for sel_duplicate_value in duplicate_values:
        row_duplicate_id = sel_dataframe[sel_col] == sel_duplicate_value
        duplicate_df = sel_dataframe[row_duplicate_id]
        new_row = {}
        for i_col in duplicate_df.columns:
            duplicate_removed = list(set(duplicate_df[i_col].values))
            duplicate_removed = [str(x) for x in duplicate_removed]
            new_row[i_col] = ', '.join(duplicate_removed)

        sel_dataframe = sel_dataframe.drop(duplicate_df.index)
        new_row_df = pd.DataFrame([new_row])
        sel_dataframe = pd.concat([sel_dataframe, new_row_df])
    return sel_dataframe.reset_index(drop=True)