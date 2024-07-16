import os
import nibabel
import numpy as np
import helper.misc as hmisc
import glob

"""
M&M-1 data has multiple cardiac phases, but only 2 segmentations

Lets split it...
"""

vendor = 'Vendor_A'
dimg = f'/data/cmr7t3t/mms1/all_phases_mid/{vendor}/Image'
ddest_img = f'/data/cmr7t3t/mms1/all_phases_mid/{vendor}/Image_single_slice'
dlabel = f'/data/cmr7t3t/mms1/all_phases_mid/{vendor}/Label'
ddest_label = f'/data/cmr7t3t/mms1/all_phases_mid/{vendor}/Label_single_slice'

file_list = os.listdir(dimg)

# Create dest dir if not existing
if os.path.isdir(ddest_img):
    files = glob.glob(ddest_img + '/*')
    [os.remove(f) for f in files]
else:
    os.makedirs(ddest_img)

if os.path.isdir(ddest_label):
    files = glob.glob(ddest_label + '/*')
    [os.remove(f) for f in files]
else:
    os.makedirs(ddest_label)

for sel_file in file_list:
    base_name = hmisc.get_base_name(sel_file)
    ext = hmisc.get_ext(sel_file)
    # Define loading  paths
    img_path = os.path.join(dimg, sel_file)
    label_path = os.path.join(dlabel, sel_file)
    # Load the image
    img_array = hmisc.load_array(img_path).T[:, ::-1, ::-1]
    # Load the segmentation
    label_array = hmisc.load_array(label_path).T[:, ::-1, ::-1]
    # Find which indices have a segmentation..
    segmented_indices = np.argwhere(label_array.sum(axis=(-2, -1)) > 0).ravel()
    for ii, i_loc in enumerate(segmented_indices):
        print('Found location ', ii, '/', len(segmented_indices), end='\r')
        # Define target paths
        # The added 0000 is for the nnUnet framework..
        img_dest_path = os.path.join(ddest_img, base_name + f'_loc_{str(i_loc).zfill(2)}' + ext)
        label_dest_path = os.path.join(ddest_label, base_name + f'_loc_{str(i_loc).zfill(2)}' + ext)
        # Store img array..
        nibabel_obj = nibabel.Nifti1Image(img_array[i_loc].T[::-1, ::-1], np.eye(4))
        nibabel.save(nibabel_obj, img_dest_path)
        # Store label
        nibabel_obj = nibabel.Nifti1Image(label_array[i_loc].T[::-1, ::-1], np.eye(4))
        nibabel.save(nibabel_obj, label_dest_path)
        print('')
