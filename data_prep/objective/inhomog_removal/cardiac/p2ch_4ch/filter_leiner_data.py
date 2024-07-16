
"""

Get valid patient list

Inspect images

Save images in different location
"""

import re
import os
import pydicom
import numpy as np
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import sys
import matplotlib.pyplot as plt

target_dir = '/home/bugger/Documents/data/1.5T'
# This one will be mapped on the findings of the target study
target_sub_dir = ['p2ch', 'p4ch', 'sa']


"""
Get a valid patient list
"""

patient_data_dir = '/media/bugger/MyBook/data/leiner_data'
dest_dir = '/home/bugger/Documents/data/1.5T'
patient_list = os.listdir(patient_data_dir)
# We want these names to be present in the study name
target_study_name = ['.*sBTFE 4k.*', '.*sBTFE 2k.*', '.*sBTFE ka.*']
re_study_name = [re.compile(x) for x in target_study_name]
dest_sub_dir = ['p4ch', 'p2ch', 'sa']
orig_dest_dict = dict(zip(re_study_name, dest_sub_dir))

N_patients = len(patient_list)
N_target_study = len(target_study_name)

study_list = []
for i in patient_list:
    temp_dir = os.path.join(patient_data_dir, i, 'Mri Hart')
    temp_studies = os.listdir(temp_dir)
    study_list.append(temp_studies)

binary_overview = np.zeros((N_patients, N_target_study))
for i, i_patient in enumerate(patient_list):
    print(i_patient)
    temp_list = study_list[i]
    temp_binary = []
    for i_study in temp_list:
        # temp = [True if i_target_name in i_study else False for i_target_name in target_study_name]
        temp = [True if x.findall(i_study) else False for x in re_study_name ]
        temp_binary.append(temp)
    result_presence = np.stack(temp_binary).sum(axis=0)
    binary_overview[i, :] = result_presence

plt.imshow(binary_overview, vmin=0, vmax=1)

# Select the patients..
patient_bin_list = [any(x) for x in binary_overview]
index_patient_sel = np.where(patient_bin_list)[0]
patient_sel_list = np.array(patient_list)[index_patient_sel]

difference_patient_list = set(patient_list).difference(set(patient_sel_list))
print('We are missing these.. \n', difference_patient_list)

"""
Inspect the images from the selected patients
"""

overview_shapes = []
for i_patient in patient_sel_list:
    print()
    print(i_patient)
    patient_id = re.findall('([0-9]+)', i_patient)[0]
    main_data_dir = os.path.join(patient_data_dir, i_patient, 'Mri Hart')
    main_data_files = os.listdir(main_data_dir)

    # Invoke the filtering on the target study names..
    for i_re_study in re_study_name:
        temp_sel_dir = [i_re_study.findall(x)[0] for x in main_data_files if i_re_study.findall(x)]
        temp_dest_sub_dir = orig_dest_dict[i_re_study]
        temp_dest_dir = os.path.join(dest_dir, temp_dest_sub_dir)

        for n_images, sel_dir in enumerate(temp_sel_dir):
            dest_file = os.path.join(temp_dest_dir, patient_id + '_' + str(n_images))
            data_dir = os.path.join(main_data_dir, sel_dir)
            file_dir = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
            A_list = [pydicom.read_file(x).pixel_array for x in file_dir]

            print('\n', sel_dir)
            shape_list = []
            img_list = []
            for x in A_list:
                temp_shape = x.shape
                if temp_shape in shape_list:
                    index_shape = shape_list.index(temp_shape)
                else:
                    index_shape = len(shape_list)
                    img_list.append([])
                    shape_list.append(temp_shape)
                    print('New shape:  ', temp_shape)
                    overview_shapes.append((temp_dest_sub_dir, temp_shape))

                img_list[index_shape].append(x)

            if len(img_list) == 1:
                A_temp = np.stack(img_list[0])
                print('Saving to ')
                print(dest_file)
                np.save(dest_file, A_temp)
            else:
                print('We have more images than we expected')
