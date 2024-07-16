import numpy as np
import os
import json
import csv
import re

"""
N4 algorithm
Single channel  indirect network
Single channel direct network
Multi channel  indirect network
Multi channel  direct network
"""


# This is a helper function to get a string with mean and std of an array
def get_mean_std_str(x, n_dec=2):
    x = np.array(x)
    x_mean = np.round(x.mean(), n_dec)
    x_std = np.round(x.std(), n_dec)
    return str(x_mean) + "±" + str(x_std)


def collect_glcm(ddata, file_name, temp_dict=None, key='none'):
    if temp_dict is None:
        temp_dict = {}
    file_list = os.listdir(ddata)
    # Relative change in GLCM feature per mask type...?
    filter_json = [x for x in file_list if x.endswith('json') and file_name in x]
    if len(filter_json):
        for i_file in filter_json:
            json_file = os.path.join(ddata, i_file)
            with open(json_file, 'r') as f:
                json_line = f.read()
            metric_dict = json.loads(json_line)
            temp_dict[key] = metric_dict
    return temp_dict


def print_dataframe(x):
    print('\t',  ' '.join([str(x) for x in x.columns]))
    for ii, irow in x.iterrows():
        print(irow.name, ' '.join([str(x) for x in irow.values]))


def collect_a_csv_file(ddata, file_name, temp_dict=None, key='test'):
    if temp_dict is None:
        temp_dict = {}
    file_list = os.listdir(ddata)
    filter_txt = [x for x in file_list if x.endswith('csv') and file_name in x]
    if len(filter_txt):
        for i_file in filter_txt:
            dcsv = os.path.join(ddata, i_file)
            with open(dcsv, 'r') as f:
                csv_obj = csv.reader(f)
                csv_content = [[float(x) for x in i_row] for i_row in csv_obj]
            temp_csv_content = np.array(csv_content).T
            temp_dict[key] = temp_csv_content
    return temp_dict


def collect_a_npy_file(ddata, file_name, temp_dict=None, key='test'):
    if temp_dict is None:
        temp_dict = {}
    file_list = os.listdir(ddata)
    filter_npy = [x for x in file_list if x.endswith('npy') and file_name in x]
    if len(filter_npy):
        for i_file in filter_npy:
            temp_value = np.load(os.path.join(ddata, i_file))
            temp_dict[key] = temp_value
    return temp_dict
