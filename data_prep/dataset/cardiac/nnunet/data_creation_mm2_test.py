import shutil
import os
import re
import helper.misc as hmisc
import pandas
import nibabel
import numpy as np

"""


"""

ddata = '/data/seb/data/mm_segmentation'
ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task997_mm2'
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=())
ddest_img = os.path.join(ddest, 'imagesTs')
ddest_label = os.path.join(ddest, 'labelsTs')

for subdir in ['test', 'train', 'validation']:
    sel_subdir = os.path.join(ddata, subdir, 'target')
    file_list = os.listdir(sel_subdir)
    file_list = [x for x in file_list if 'SA' in x]
    for label_file in file_list:
        img_file = re.sub('_gt', '', label_file)
        base_name = hmisc.get_base_name(img_file)
        base_ext = hmisc.get_ext(img_file)
        #
        source_img_path = os.path.join(ddata, subdir, 'input', img_file)
        source_label_path = os.path.join(sel_subdir, label_file)
        #
        target_img_path = os.path.join(ddest, 'imagesTs', base_name + "_0000" + base_ext)
        target_label_path = os.path.join(ddest, 'labelsTs', img_file)
        # print('From', source_img_path, 'to', target_img_path)
        # print('From', source_label_path, 'to', target_label_path)
        source_img_array = hmisc.load_array(source_img_path)
        source_label_array = hmisc.load_array(source_label_path)
        label_index_1 = source_label_array == 1
        label_index_3 = source_label_array == 3
        source_label_array[label_index_1] = 3
        source_label_array[label_index_3] = 1
        n_loc = source_img_array.shape[-1]
        sel_loc = n_loc // 2
        affine_array = nibabel.load(source_img_path).affine
        nibabel_obj = nibabel.Nifti1Image(source_img_array[:, :, sel_loc:sel_loc+1], affine_array)
        nibabel.save(nibabel_obj, target_img_path)
        nibabel_obj = nibabel.Nifti1Image(source_label_array[:, :, sel_loc:sel_loc+1], affine_array)
        nibabel.save(nibabel_obj, target_label_path)


