import helper.misc as hmisc
import re
import shutil
import itertools
import collections
import numpy as np
import os
import sys
from nnunet.paths import nnUNet_raw_data
from objective_configuration.segment7T3T import TASK_NR_TO_DIR


def remove_wrong_data(label_dir, image_dir):
    file_list = [os.path.join(label_dir, x) for x in os.listdir(label_dir)]
    resulting_count = [load_and_get_mid_segm(x) for x in file_list]
    print('class counter', collections.Counter(resulting_count))
#
    files_to_be_deleted_2 = list(itertools.chain(*np.array(file_list)[np.argwhere(np.array(resulting_count) == 2)]))
    files_to_be_deleted_1 = list(itertools.chain(*np.array(file_list)[np.argwhere(np.array(resulting_count) == 1)]))
#
    for i_file in files_to_be_deleted_2 + files_to_be_deleted_1:
        file_name = os.path.basename(i_file)
        sel_label_file = os.path.join(label_dir, file_name)
        sel_image_file = os.path.join(image_dir, file_name)
        if os.path.isfile(sel_label_file):
            os.remove(sel_label_file)
        else:
            print('not found', sel_label_file)
        if os.path.isfile(sel_image_file):
            os.remove(sel_image_file)
        else:
            print('not found', sel_image_file)



def load_and_get_mid_segm(sel_file):
    sel_array = hmisc.load_array(sel_file)
    ind_class_2 = sel_array == 2
    index_where = np.argwhere(ind_class_2)
    avg_index = index_where.mean(axis=0).astype(int)
    x0, y0, z0 = avg_index
    LV_value = sel_array[x0, y0, z0]
    return LV_value

"""
We messed it up...
"""

task_number = '501'
#for task_number in ['501', '502', '630', '631', '633', '635'] + [str(x) for x in range(901, 910)]:
for task_number in ['633', '635']:
    print(task_number)
    task_dir = TASK_NR_TO_DIR.get(task_number.zfill(3), 'Unknown')
    label_train = os.path.join(nnUNet_raw_data, task_dir, 'labelsTr')
    label_test = os.path.join(nnUNet_raw_data, task_dir, 'labelsTs')
    image_train = os.path.join(nnUNet_raw_data, task_dir, 'imagesTr')
    image_test = os.path.join(nnUNet_raw_data, task_dir, 'imagesTs')
#
    # Lets first test it...
    remove_wrong_data(label_dir=label_train, image_dir=image_train)
    remove_wrong_data(label_dir=label_test, image_dir=image_test)


def check_overlap(task_num):
    task_num = str(task_num)
    task_dir = TASK_NR_TO_DIR.get(task_num.zfill(3), 'Unknown')
    label_train = os.path.join(nnUNet_raw_data, task_dir, 'labelsTr')
    label_test = os.path.join(nnUNet_raw_data, task_dir, 'labelsTs')
    image_train = os.path.join(nnUNet_raw_data, task_dir, 'imagesTr')
    image_test = os.path.join(nnUNet_raw_data, task_dir, 'imagesTs')
    label_train_files = os.listdir(label_train)
    image_train_files = [re.sub('_0000', '', x) for x in os.listdir(image_train)]
    label_test_files = os.listdir(label_test)
    image_test_files = [re.sub('_0000', '', x) for x in os.listdir(image_test)]
    missing_files = get_missing_stuff(label_train_files, image_train_files)
    remove_missing_stuff(missing_files, task_dir=task_dir, subdir='imagesTr')
    missing_files = get_missing_stuff(label_test_files, image_test_files)
    remove_missing_stuff(missing_files, task_dir=task_dir, subdir='imagesTs')

def get_missing_stuff(label_files, image_files):
    print('Example images ', label_files[:10])
    print('Example images ', image_files[:10])
    print('Train label \ Train image', set(label_files).difference(set(image_files)))
    img_min_label_files = list(set(image_files).difference(set(label_files)))
    print('Train image \ Train label ', len(img_min_label_files))
    return img_min_label_files

def remove_missing_stuff(missing_files, task_dir, subdir):
    for i_file in missing_files:
        base_name = hmisc.get_base_name(i_file)
        base_ext = hmisc.get_ext(i_file)
        orig_img_file = os.path.join(nnUNet_raw_data, task_dir, subdir, base_name + '_0000' + base_ext)
        if os.path.isfile(orig_img_file):
            os.remove(orig_img_file)
        else:
            print('Couldnt find ', orig_img_file)


check_overlap(630)
check_overlap(631)
check_overlap(633)
check_overlap(635)