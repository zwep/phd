import hashlib
import re
import json
import os
import argparse

"""
Here we hope to get some md5sum output..
This can then be run remotely on both machines

Later the output of both files can be compared.
"""

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str)
p_args = parser.parse_args()
directory = p_args.d
directory_str = re.sub('\\/', '_', directory)
md5_json = os.path.expanduser(f'~/md5sum{directory_str}.json')

os.path.relpath('/home/bugger/Documents/data', '/home/bugger/Documents')
hd5sum_dict = {}
for d, _, f in os.walk(directory):
    if len(f):
        for i_file in f:
            file_path = os.path.join(d, i_file)
            rel_file_path = os.path.relpath(file_path, start=directory)
            try:
                file_content = open(file_path, 'rb').read()
                md5hex = hashlib.md5(file_content).hexdigest()
                hd5sum_dict[rel_file_path] = md5hex
            except IOError:
                print('IOError for file ', rel_file_path)
                hd5sum_dict[rel_file_path] = 'IOError'

serialized_json = json.dumps(hd5sum_dict)
with open(md5_json, 'w') as f:
    f.write(serialized_json)
