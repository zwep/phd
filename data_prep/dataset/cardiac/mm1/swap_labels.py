import os
import nibabel
import helper.misc as hmisc
import glob

"""

# This is how we interpret values and classes...
# class_interpretation = {'1': 'RV', '2': 'MYO', '3': 'LV'}
# But this is how MM1 has it..
# class_interpretation = {'3': 'RV', '2': 'MYO', '1': 'LV'}
# So we need to switch them
"""

vendor = 'Vendor_B'
dlabel = f'/data/cmr7t3t/mms1/all_phases_mid/{vendor}/Label_single_slice'
dlabel_dest = f'/data/cmr7t3t/mms1/all_phases_mid/{vendor}/Label_single_slice_swapped'

if os.path.isdir(dlabel_dest):
    files = glob.glob(dlabel_dest + '/*')
    [os.remove(f) for f in files]
else:
    os.makedirs(dlabel_dest)

for i_file in os.listdir(dlabel):
    dlabel_file = os.path.join(dlabel, i_file)
    dlabel_file_dest = os.path.join(dlabel_dest, i_file)
    nibabel_obj = nibabel.load(dlabel_file)
    label_array = nibabel_obj.get_fdata()
    label_index_1 = label_array == 1
    label_index_3 = label_array == 3
    label_array[label_index_1] = 3
    label_array[label_index_3] = 1
    new_obj = nibabel.Nifti1Image(label_array, nibabel_obj.affine)
    nibabel.save(new_obj, dlabel_file_dest)
