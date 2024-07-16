from objective_configuration.segment7T3T import CLASS_INTERPRETATION, COLOR_DICT, \
    COLOR_DICT_RGB, CMAP_ALL, MY_CMAP, get_path_dict
import nibabel
import os
import helper.misc as hmisc

"""
Lets check how big the problem is

"""

for idataset in ['acdc', '7t', 'mm1a', 'mm1b']:
    img_dir = get_path_dict(idataset)['dimg']
    sel_file = os.listdir(img_dir)[0]
    sel_file_dir = os.path.join(img_dir, sel_file)
    loaded_array = nibabel.load(sel_file_dir)
    print(loaded_array.affine)




