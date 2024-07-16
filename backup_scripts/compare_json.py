import json
import os
import argparse


def load_json(file_path):
    with open(file_path, 'r') as f:
        temp = f.read()
    temp_json = json.loads(temp)
    return temp_json

"""
With this file we can compare the output given in both json files
"""


parser = argparse.ArgumentParser()
parser.add_argument('-file_a', type=str)
parser.add_argument('-file_b', type=str)
p_args = parser.parse_args()
json_file_a = p_args.file_a
json_file_b = p_args.file_b

json_obj_a = load_json(json_file_a)
json_obj_b = load_json(json_file_b)

set_a_keys = set(json_obj_a.keys())
set_b_keys = set(json_obj_b.keys())

difference_a_b_keys = set_a_keys.difference(set_b_keys)
difference_b_a_keys = set_b_keys.difference(set_a_keys)
if len(difference_a_b_keys) or len(difference_b_a_keys):
    print("The following files difference exist")
    print("Exist in A, not in B ", difference_a_b_keys)
    print("Exist in B, not in A ", difference_a_b_keys)
else:
    print('No difference in file names has been found\n\n')

set_a_b_keys = set_a_keys.intersection(set_b_keys)
md5_comparisson = {}
for i_key in set_a_b_keys:
    md5_comparisson.setdefault(i_key, False)
    if json_obj_a[i_key] == json_obj_b[i_key]:
        md5_comparisson[i_key] = True

# Print which comparissons turn out to be false..
for k, v in md5_comparisson.items():
    if v is False:
        print('False md5sum ', k)

