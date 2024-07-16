
import json
import os

ddir_json = '/home/bugger/Documents/model_run/inhomog_clean_list'
dict_replace = [(['data', 'target_type'], 'rho', 'biasfield')]

list_json = []
for i_file in os.listdir(ddir_json):
    path_json = os.path.join(ddir_json, i_file)
    with open(path_json, 'r') as f:
        A = json.loads(f.read())

    list_json.append([A, path_json])

import helper.misc as hmisc
for i_key, i_old, i_new in dict_replace:
    for i_dict, _ in list_json:
        res_values = hmisc.get_nested(i_dict, i_key)
        print(i_key, res_values)

