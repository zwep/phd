import pydicom
import os
import numpy as np

ddata_1 = '/media/bugger/MyBook/data/7T_scan/phantom/2021_02_12_Copper_Wire_Straight/DICOM'
ddata_2 = '/media/bugger/MyBook/data/7T_scan/phantom/2021_08_18_Wire_40cm/DICOM'
ddata_3 = '/media/bugger/MyBook/data/7T_scan/phantom/2021_08_27_Wire_40cm_Side/DICOM'

import re
re_obj = re.compile('CLEAR_FA[0-9]{2}.dcm')
sel_file_1 = sorted([os.path.join(ddata_1, x) for x in os.listdir(ddata_1) if re_obj.findall(x)])
dicom_obj_1 = pydicom.read_file(sel_file_1[0], stop_before_pixels=True)


sel_file_2 = sorted([os.path.join(ddata_2, x) for x in os.listdir(ddata_2) if re_obj.findall(x)])
# Hackish
faulty_index = [i for i, x in enumerate(sel_file_2) if 'FA08' in x][0]
sel_file_2.pop(faulty_index)

dicom_obj_2 = pydicom.read_file(sel_file_2[0], stop_before_pixels=True)

sel_file_3 = sorted([os.path.join(ddata_3, x) for x in os.listdir(ddata_3) if re_obj.findall(x)])
# Hackish
faulty_index = [i for i, x in enumerate(sel_file_3) if 'FA08' in x][0]
sel_file_3.pop(faulty_index)
dicom_obj_3 = pydicom.read_file(sel_file_3[0], stop_before_pixels=True)

import helper.misc as hmisc
def compare_two_dicoms(dicom_obj_1, dicom_obj_2):
    key_1 = set(dicom_obj_1.keys())
    key_2 = set(dicom_obj_2.keys())
    different_keys = []
    for i_key in key_1.intersection(key_2):
        bool_int = isinstance(dicom_obj_1[i_key].value, int)
        bool_float = isinstance(dicom_obj_1[i_key].value, float)
        bool_str = isinstance(dicom_obj_1[i_key].value, str)
        if bool_str or bool_int or bool_float:
            if dicom_obj_1[i_key].value != dicom_obj_2[i_key].value:
                different_keys.append(i_key)

    for i_key in different_keys:
        print(dicom_obj_1[i_key].description(), dicom_obj_1[i_key].tag, dicom_obj_1[i_key].value, '---', dicom_obj_2[i_key].value)

    return different_keys

print('file 1', os.path.dirname(sel_file_1[0]))
print('file 2', os.path.dirname(sel_file_2[0]))
print('file 3', os.path.dirname(sel_file_3[0]))
for x_file, y_file in zip(sel_file_1, sel_file_3):
    print('\n\n\n')
    print('Comparing ', os.path.basename(x_file), '---', os.path.basename(y_file))
    dicom_obj_y = pydicom.read_file(y_file, stop_before_pixels=True)
    dicom_obj_x = pydicom.read_file(x_file, stop_before_pixels=True)
    # _ = compare_two_dicoms(dicom_obj_x, dicom_obj_y)

    print('x', [dicom_obj_x[('2005', '142a')], dicom_obj_x[('2005', '142b')],dicom_obj_x[('2005', '142c')],dicom_obj_x[('2005', '142d')],dicom_obj_x[('2005', '142e')],dicom_obj_x[('2005', '142f')]])
    print('y', [dicom_obj_y[('2005', '142a')], dicom_obj_y[('2005', '142b')],dicom_obj_y[('2005', '142c')],dicom_obj_y[('2005', '142d')],dicom_obj_y[('2005', '142e')],dicom_obj_y[('2005', '142f')]])
