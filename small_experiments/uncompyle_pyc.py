
"""
Because uncompyle2 has no recursive option.. we are going to fix it in python
"""

import re
import os
import shutil

raw_loader_dir = '/home/bugger/Documents/philips_data_readers'

possible_loader = [x for x in os.listdir(raw_loader_dir) if 'ReadPhilips_' in x and not x.endswith('zip')]
possible_loader = [x for x in possible_loader if '2015-04-03' not in x]
# uncompile_program = 'uncompyle2'
uncompile_program = 'uncompyle6'

for load_dir in possible_loader:
    walk_dir = os.path.join(raw_loader_dir, load_dir, 'philips')
    dest_dir = os.path.join(raw_loader_dir, load_dir, 'philips_py')
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)

    os.mkdir(dest_dir)

    for d, _, f in os.walk(walk_dir):
        sel_files = [x for x in f if x.endswith('pyc')]
        for i_file in sel_files:
            source_file = os.path.join(d, i_file)
            target_dir = os.path.dirname(re.sub('/philips/', '/philips_py/', source_file))
            target_file = os.path.join(target_dir, os.path.splitext(i_file)[0] + '.py')
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)

            exec_string = uncompile_program + ' -o ' + target_file + ' ' + source_file
            os.system(exec_string)
