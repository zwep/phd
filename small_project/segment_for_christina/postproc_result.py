import os
import re
import numpy as np
import nibabel
import helper.misc as hmisc

"""
The results are 3D arrays.. lets recombine them to 4D..
"""


dsource = '/data/seb/data/mm_christina/original_image'
dsegm = '/data/seb/data/mm_christina/segm_result'
dtarget = '/data/seb/data/mm_christina/postproc_segm_result'

file_list_source = os.listdir(dsource)
file_list_segm = os.listdir(dsegm)

for i_file in file_list_source:
    file_name = hmisc.get_base_name(i_file)
    # Get the struct..
    source_file = os.path.join(dsource, i_file)
    source_struct = nibabel.load(source_file).affine
    # Define target file
    target_file = os.path.join(dtarget, i_file)
    # Load all the created segm files and sort them based on their phase/location
    filter_file_segm = [x for x in file_list_segm if file_name in x]
    filter_file_segm = sorted(filter_file_segm, key=lambda x: int(re.findall('sa_([0-9]+)', x)[0]))
    # Load and concat everything
    full_segm_array = []
    for i_segm in filter_file_segm:
        segm_file = os.path.join(dsegm, i_segm)
        segm_array = hmisc.load_array(segm_file).T
        full_segm_array.append(segm_array)
    #
    full_segm_array = np.array(full_segm_array)
    print('Final shape ', full_segm_array.shape)
    # Convert to nifti... and save
    nibabel_obj = nibabel.Nifti1Image(full_segm_array.T, source_struct)
    nibabel.save(nibabel_obj, target_file)
