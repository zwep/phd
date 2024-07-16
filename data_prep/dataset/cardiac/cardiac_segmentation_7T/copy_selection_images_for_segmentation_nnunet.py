"""
We all selected our favourite SA images (7T)

I take the union of all this. Make that as 'the set'

I want to make ground turth labels for these 7T images. Initialize these things iwth NNunet
SO lets move them over to a nice directory with proper names
"""

# # Copy some data from Sina over to my side with a different name...
# The name change is needed to comply with the naming convention of NNunet
import os
import shutil
import re

source_dir_seb = '/data/cmr7t3t/cmr7t/Image_selection_seb'
source_dir_sina = '/data/cmr7t3t/cmr7t/Image_selection_sina'
source_dir_yasmina = '/data/cmr7t3t/cmr7t/Image_selection_yasmina'

set_yasmina = set(os.listdir(source_dir_yasmina))
set_seb = set(os.listdir(source_dir_seb))
set_sina = set(os.listdir(source_dir_sina))

source_files = list(set_sina.union(set_seb).union(set_yasmina))
source_dir_img = '/data/cmr7t3t/cmr7t/Image_all'
dest_dir_img = '/data/cmr7t3t/cmr7t/Image'

# Copy the union of all our selections to a new folder
for i_file in sorted(source_files):
    old_loc = os.path.join(source_dir_img, i_file)
    new_loc = os.path.join(dest_dir_img, i_file)
    shutil.copy(old_loc, new_loc)


"""
Now move the select in the ./Image folder to the folder where we can process it with nnunet
"""

source_dir_nnunet = '/data/cmr7t3t/cmr7t/Image_all'
dest_dir_nnunet = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task999_7T/imagesTs'
# Here we store the name changes so that we can change it back easily
dest_dir_name_conv = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task999_7T/sina2nnunet_names.json'

name_conv_dict = {}
for i_file in sorted(source_files):
    old_loc = os.path.join(source_dir_nnunet, i_file)
    find_loc = re.findall("_([0-9]).nii.gz", i_file)[0]
    find_pat = re.findall("_([0-9]{4})_", i_file)[0]
    find_pat = list(find_pat)
    find_pat[1] = find_loc
    find_pat = ''.join(find_pat)
    new_file = re.sub("_[0-9].nii.gz", "_0000.nii.gz", i_file)
    new_file = re.sub("_[0-9]{4}_", f"_{find_pat}_", new_file)
    print(i_file, new_file)
    name_conv_dict[i_file] = new_file
    new_loc = os.path.join(dest_dir_nnunet, new_file)
    shutil.copy(old_loc, new_loc)

import json
serialized_dict = json.dumps(name_conv_dict)
with open(dest_dir_name_conv, 'w') as f:
    f.write(serialized_dict)