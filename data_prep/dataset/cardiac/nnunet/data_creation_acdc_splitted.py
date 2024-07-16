import os
import helper.misc as hmisc
import nibabel
import sys
import shutil

"""
Copy data from our directory to a new challenge.
"""

dtarget = "/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task511_ACDC"
dsource = "/data/cmr7t3t/acdc/acdc_processed"
mapping_of_folders = {"Image": "imagesTr", "ImageTest": "imagesTs", "Label": "labelsTr", "LabelTest": "labelsTs"}

for i_source, i_target in mapping_of_folders.items():
    source_dir = os.path.join(dsource, i_source)
    target_dir = os.path.join(dtarget, i_target)
    source_files = os.listdir(source_dir)

    for i_source_file in source_files:
        # Perform name change..
        if 'Image' in i_source:
            i_source_name = hmisc.get_base_name(i_source_file)
            i_source_ext = hmisc.get_ext(i_source_file)
            i_target_file = i_source_name + "_0000" + i_source_ext
        else:
            i_target_file = i_source_file

        source_path = os.path.join(source_dir, i_source_file)
        target_path = os.path.join(target_dir, i_target_file)
        print("Copying ", source_path, "  --- to --- ", target_path)
        shutil.copy(source_path, target_path)
