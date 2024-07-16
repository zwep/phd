
import helper.plot_class as hplotc
from skimage.util import img_as_ubyte, img_as_uint
import re
import pydicom
import skimage.transform as sktransform
import os
import helper.misc as hmisc
import SimpleITK as sitk
import helper.array_transf as harray
import objective.inhomog_removal.executor_inhomog_removal as executor
import os
import json
import torch
import numpy as np
import glob

"""
We need some conventional models on the patient data set..
"""

# Load some models

import biasfield_algorithms.N4ITK as model_n4itk
import biasfield_algorithms.BBHE as model_bbhe

# fun_list = [model_HF.get_hf, model_HF.get_hf_l0, model_pabic.get_lst_sqr, model_n4itk.get_n4itk, model_hum.get_holomorfic, model_bbhe.get_bbhe, None]
fun_list = [model_n4itk.get_n4itk, model_bbhe.get_bbhe]

# Load some data

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
# Select only one patient (for which we have created a shit ton of masks
for i_patient in patient_id_list:

    print(i_patient)

    t2w_file_list_filter = [x for x in t2w_file_list if i_patient in x and x.endswith('dcm')]
    counter = 0
    result_list = []
    sel_file = t2w_file_list_filter[counter]
    # Some patients have 1 file, others have multiple...
    # print(len(t2w_file_list_filter))
    """
    Loop over all the T2w files for this patient
    """
    for sel_file in t2w_file_list_filter:
        file_name = os.path.basename(sel_file)
        file_name, _ = os.path.splitext(file_name)

        target_file = os.path.join(target_dir, file_name + '.dcm')
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        pydicom_obj = pydicom.read_file(sel_file)
        acq_nr = pydicom_obj.get(('0020', '0012'))
        instance_nr = pydicom_obj.get(('0020', '0013'))
        # print('\t, ', acq_nr, instance_nr)

        pixel_array = pydicom_obj.pixel_array
        # print('Array size', pixel_array.shape, sel_file)
        ndim = pixel_array.ndim
        if ndim == 3:
            n_slice = pixel_array.shape[0]
        else:
            n_slice = 1
            pixel_array = pixel_array[None]

        mask_file_name = os.path.splitext(sel_file)[0] + "_mask.npy"
        if os.path.isfile(mask_file_name):
            mask_array = np.load(mask_file_name)

        if (mask_array.sum() == 0) or (not os.path.isfile(mask_file_name)):
            if pixel_array.ndim == 2:
                print('Shape pixel array ', pixel_array.shape)
                initial_mask = harray.get_treshold_label_mask(pixel_array)
                mask_array, verts = harray.convex_hull_image(initial_mask)
            else:
                print('Number of dimensions is now.. ', ndim)
                initial_mask = [harray.get_treshold_label_mask(x) for x in pixel_array]
                mask_array, verts = zip(*[harray.convex_hull_image(x) for x in initial_mask])
                mask_array = np.array(mask_array)

            np.save(mask_file_name, mask_array)

        if mask_array.ndim == 2:
            mask_array = mask_array[None]

        """
        For those images where we have multiple slices.. loop over them
        """

        for i in range(n_slice):
            total_counter += 1
            print('Performing slice', i)
            pixel_array_slice = pixel_array[i]
            mask_array_slice = mask_array[i]

# Evaluate them...
