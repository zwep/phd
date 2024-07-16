
"""
We got some remote results

We want them in the correct naming convention

Here we have a script that changes a folder with my nnunet naming convention to the one used by sina et al


The results from the segmentations were copied from /data/seb/result_acdc
The images used by nnunet are stored in /data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task999_7T
"""

import argparse
import os
import json
import helper.misc as hmisc
import getpass

if getpass.getuser() == 'bugger':
    name_change_location = '/home/bugger/Documents/data/7T/cardiac/segment7T3T/sina2nnunet_names.json'
else:
    name_change_location = '/data/seb/data/sina2nnunet_names.json'

with open(name_change_location, 'r') as f:
    name_change_dict = json.loads(f.read())

name_change_dict = {v: k for k, v in name_change_dict.items()}


def change_file_name(ddir, ext=None):
    for i_file in os.listdir(ddir):
        # The results from nnunet omit the _0000 (or any other modality appendix). Which is logical
        # However, I have it in my file name conversion file..... so I need to add it here....
        base_name = hmisc.get_base_name(i_file)
        file_ext = hmisc.get_ext(i_file)
        if ext == file_ext:
            base_name = base_name + "_0000" + file_ext
            new_name = name_change_dict[base_name]
            source_file = os.path.join(ddir, i_file)
            dest_file = os.path.join(ddir, new_name)
            print(source_file, dest_file)
            os.rename(source_file, dest_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str)
    p_args = parser.parse_args()
    model_name = p_args.model_name
    ddir_initial_labels = f'/data/cmr7t3t/cmr7t/Results/{model_name}'