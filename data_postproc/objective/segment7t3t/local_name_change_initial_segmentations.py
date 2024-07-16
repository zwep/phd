"""
Okay

used this for the images

scp -r seb@legolas.bmt.tue.nl:/data/cmr7t3t/cmr7t/Image/ /home/bugger/Documents/data/7T/cardiac/segment7T3T


Did the segmentations (initial, reuslts from nnunet) by hand

Hoooweever. These results from nnunet haave different names than those from the project.
Change them here (from nnunet names to project names)
(used locally)
"""

import os
import json
import helper.misc as hmisc

ddir_initial_labels = '/home/bugger/Documents/data/7T/cardiac/segment7T3T/initial_labels'
name_change_location = '/home/bugger/Documents/data/7T/cardiac/segment7T3T/sina2nnunet_names.json'

with open(name_change_location, 'r') as f:
    name_change_dict = json.loads(f.read())

name_change_dict = {v: k for k, v in name_change_dict.items()}

for i_file in os.listdir(ddir_initial_labels):
    # The results from nnunet omit the _0000 (or any other modality appendix). Which is logical
    # However, I have it in my file name conversion file.....
    base_name = hmisc.get_base_name(i_file)
    file_ext = hmisc.get_ext(i_file)
    base_name = base_name + "_0000" + file_ext

    new_name = name_change_dict[base_name]
    source_file = os.path.join(ddir_initial_labels, i_file)
    dest_file = os.path.join(ddir_initial_labels, new_name)
    os.rename(source_file, dest_file)
