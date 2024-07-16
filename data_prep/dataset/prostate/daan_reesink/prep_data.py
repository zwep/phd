import re
import pydicom
import collections
import helper.array_transf as harray
import os
import numpy as np


def create_mask(x):
    if x.ndim == 2:
        initial_mask = harray.get_treshold_label_mask(x)
        mask_array, verts = harray.convex_hull_image(initial_mask)
    else:
        initial_mask = [harray.get_treshold_label_mask(x) for x in x]
        mask_array, verts = zip(*[harray.convex_hull_image(x) for x in initial_mask])
        mask_array = np.array(mask_array)

    if mask_array.ndim == 2:
        mask_array = mask_array[None]

    return mask_array


"""
Some dicom files are multiple files.. others are a single 3D file
Here we homogenize that stuff
"""

ddata = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/DaanReesink'
ddest_img = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/img'
ddest_mask = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/mask'

# Get the data
t2w_file_list = []
for d, _, f in os.walk(ddata):
    filter_f = [x for x in f if 'T2w' in x]
    if len(filter_f):
        for i_file in filter_f:
            temp_file = os.path.join(d, i_file)
            t2w_file_list.append(temp_file)

patient_id_pattern = re.compile('(7TMRI[0-9]{3})')
patient_id_list = list(set([patient_id_pattern.findall(x)[0] for x in t2w_file_list if patient_id_pattern.findall(x)]))
patient_id_list = sorted(patient_id_list)

total_counter = 0
# i_patient = '7TMRI003'
for i_patient in patient_id_list:
    print(i_patient)
    t2w_file_list_filter = [x for x in t2w_file_list if i_patient in x]
    # I want to sort the files based on their acq/instance number so that we get the correct order
    n_files = len(t2w_file_list_filter)
    if n_files > 1:
        acq_instance_nr_list = []
        for sel_file in t2w_file_list_filter:
            file_name = os.path.basename(sel_file)
            file_name, _ = os.path.splitext(file_name)

            pydicom_obj = pydicom.read_file(sel_file, stop_before_pixels=True)
            acq_nr = pydicom_obj.get(('0020', '0012'))
            instance_nr = pydicom_obj.get(('0020', '0013'))
            acq_instance_nr_list.append((int(acq_nr.value), int(instance_nr.value), sel_file))

        sorted_file_list = sorted(acq_instance_nr_list, key=lambda x: (x[0], x[1]))

        most_common_acq, _ = collections.Counter([x[0] for x in acq_instance_nr_list]).most_common()[0]
        sorted_file_list = [x for x in sorted_file_list if x[0] == most_common_acq]

        pixel_array = []
        for _, _, sel_file in sorted_file_list:
            pydicom_obj = pydicom.read_file(sel_file)
            temp_pixel_array = pydicom_obj.pixel_array
            pixel_array.append(temp_pixel_array)

        pixel_array = np.array(pixel_array)
    else:
        sel_file = t2w_file_list_filter[0]
        pydicom_obj = pydicom.read_file(sel_file)
        pixel_array = pydicom_obj.pixel_array

    print("Shape of pixel array ", pixel_array.shape)
    mask_array = create_mask(x=pixel_array)
    dest_img_file = os.path.join(ddest_img, i_patient)
    dest_mask_file = os.path.join(ddest_mask, i_patient)

    np.save(dest_img_file, pixel_array)
    np.save(dest_mask_file, mask_array)