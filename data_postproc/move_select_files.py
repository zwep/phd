
import os
import re
import shutil

re_match = re.compile('^Duke')
# re_match = re.compile('^M1[0-9]')
orig_dir = '/data/seb/semireal/prostate_simulation_rxtx/train/input'
dest_dir = '/data/seb/semireal/prostate_simulation_rxtx/train_old/input'

current_files = os.listdir(orig_dir)
for i_file in current_files:
    if re_match.match(i_file) is not None:
        cur_path = os.path.join(orig_dir, i_file)
        dest_path = os.path.join(dest_dir, i_file)
        shutil.move(cur_path, dest_path)