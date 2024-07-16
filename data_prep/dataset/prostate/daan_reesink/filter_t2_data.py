"""
I manually rename the files that I want to be copied.

So I append a `_T2w` string to the name of the files that I would like to see

Since there are many other views and type of acquisitions besides the T2w axial img

"""

import itertools
import collections
import re
import pydicom
import skimage.transform as sktransform
import os
import numpy as np
import helper.array_transf as harray


def load_mask_or_create(mask_file_name, x):
    if os.path.isfile(mask_file_name):
        mask_array = np.load(mask_file_name)
    else:
        if x.ndim == 2:
            print('Shape pixel array ', x.shape)
            initial_mask = harray.get_treshold_label_mask(x)
            mask_array, verts = harray.convex_hull_image(initial_mask)
        elif x.ndim == 3:
            initial_mask = [harray.get_treshold_label_mask(x) for x in x]
            mask_array, verts = zip(*[harray.convex_hull_image(x) for x in initial_mask])
            mask_array = np.array(mask_array)
        else:
            mask_array = None
            pass

    return mask_array


ddata = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/DaanReesink'
target_dir_daan = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter'
target_dir_img = os.path.join(target_dir_daan, 'image')
target_dir_mask = os.path.join(target_dir_daan, 'mask')
if not os.path.isdir(target_dir_img):
    os.makedirs(target_dir_img)

if not os.path.isdir(target_dir_mask):
    os.makedirs(target_dir_mask)

t2w_file_list = []
for d, _, f in os.walk(ddata):
    if len(f):
        filter_f = [x for x in f if 'T2w' in x]
        if len(filter_f):
            for i_file in filter_f:
                temp_file = os.path.join(d, i_file)
                mri_id = re.findall("7TMRI[0-9]{3}", d)[0]
                t2w_file_list.append((mri_id, temp_file))

# Collect the files..
# Create masks and store the data
for patient_id, iter_files in itertools.groupby(t2w_file_list, key=lambda x: x[0]):
    file_list = list(iter_files)
    target_img_file = os.path.join(target_dir_img, patient_id + '.npy')
    target_mask_file = os.path.join(target_dir_mask, patient_id + '.npy')

    print("Patient ID", patient_id)
    print("Number of files ", len(file_list))

    # If we have multiple files....
    if len(file_list) > 1:
        # First find the right order of each individual file
        acq_instance_nr_list = []
        for _, sel_file in file_list:
            file_name = os.path.basename(sel_file)
            file_name, _ = os.path.splitext(file_name)

            pydicom_obj = pydicom.read_file(sel_file, stop_before_pixels=True)
            acq_nr = pydicom_obj.get(('0020', '0012'))
            instance_nr = pydicom_obj.get(('0020', '0013'))
            acq_instance_nr_list.append((int(acq_nr.value), int(instance_nr.value), sel_file))

        # Leave out other acq numbers that dont belong
        sorted_file_list = sorted(acq_instance_nr_list, key=lambda x: (x[0], x[1]))
        most_common_acq, _ = collections.Counter([x[0] for x in acq_instance_nr_list]).most_common()[0]
        sorted_file_list = [x for x in sorted_file_list if x[0] == most_common_acq]

        # Load everything
        pixel_array_list = []
        mask_array_list = []
        for _, _, sel_file in sorted_file_list:
            pydicom_obj = pydicom.read_file(sel_file)
            temp_pixel_array = pydicom_obj.pixel_array
            temp_mask_array = load_mask_or_create(target_mask_file, x=temp_pixel_array)
            pixel_array_list.append(temp_pixel_array)
            mask_array_list.append(temp_mask_array)

        pixel_array = np.array(pixel_array_list)
        mask_array = np.array(mask_array_list)
    else:
        _, sel_file = file_list[0]
        pixel_array = pydicom.read_file(sel_file).pixel_array
        mask_array = load_mask_or_create(target_mask_file, x=pixel_array)

    print("Dimensions ", pixel_array.shape)
    print("Dimensions ", mask_array.shape)

    np.save(target_img_file, pixel_array)
    np.save(target_mask_file, mask_array)