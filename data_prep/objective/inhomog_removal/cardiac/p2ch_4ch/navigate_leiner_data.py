"""

Search thourhg Leiner his data...

"""

import os
import pydicom
import numpy as np
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import sys
import matplotlib.pyplot as plt

"""
Plot the content of a single patient study
"""

main_data_dir = '/media/bugger/MyBook/data/leiner_data/Anonymized - 0072072/Mri Hart'

for sel_dir in os.listdir(main_data_dir):
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

        img_list[index_shape].append(x)

    for i_img_list in img_list:
        A_temp = np.stack(i_img_list)
        # Somehow need to 'discorver' the pixel dimensions
        if A_temp.shape[-1] == 3:
            A_temp = np.moveaxis(A_temp, -1, 0)
        hplotc.SlidingPlot(A_temp, title=sel_dir)


"""
Search through all the patients and see how many have a certain target study we want

Conclussion: Almost everyone has the current target study names in them
Except for patient Anonymized - 0702732
"""

patient_data_dir = '/media/bugger/MyBook/data/leiner_data'
patient_list = os.listdir(patient_data_dir)
# We want these names to be present in the study name
target_study_name = ['4k', '2k', 'ka']
N_patients = len(patient_list)
N_target_study = len(target_study_name)

study_list = []
for i in patient_list:
    temp_dir = os.path.join(patient_data_dir, i, 'Mri Hart')
    temp_studies = os.listdir(temp_dir)
    study_list.append(temp_studies)

binary_overview = np.zeros((N_patients, N_target_study))
for i_patient in range(N_patients):
    print(patient_list[i_patient])
    temp_list = study_list[i_patient]
    temp_binary = []
    for i_study in temp_list:
        temp = [True if i_target_name in i_study else False for i_target_name in target_study_name]
        temp_binary.append(temp)
    result_presence = np.stack(temp_binary).sum(axis=0)
    binary_overview[i_patient, :] = result_presence

plt.imshow(binary_overview, vmin=0, vmax=1)
# Select the patients..
patient_sel_list = [all(x) for x in binary_overview]
index_patient_sel = np.where(patient_sel_list)[0]
patient_sel = np.array(patient_list)[index_patient_sel]


"""
Search through all the patients and what overlap they have in their studies

Conclussion: There is hardly any overlap... So we cannot use any smart metric or so..
"""

patient_data_dir = '/media/bugger/MyBook/data/leiner_data'
patient_list = os.listdir(patient_data_dir)

study_list = []
for i in patient_list:
    temp_dir = os.path.join(patient_data_dir, i, 'Mri Hart')
    temp_studies = os.listdir(temp_dir)
    study_list.append(temp_studies)

N = len(study_list)
A_overlap = np.zeros((N, N))
for i_study in range(N):
    temp_study = study_list[i_study]
    for i_counter_study in range(i_study, N):
        temp_counter_study = study_list[i_counter_study]

        set_study = set(temp_study)
        N_shots = len(set_study)
        set_counter_study = set(temp_counter_study)
        N_overlap = len(set_study.union(set_counter_study))
        percentage_overlap = N_shots/N_overlap
        A_overlap[i_study, i_counter_study] = percentage_overlap

plt.imshow(A_overlap)

"""
Search through all the dicoms for the tag of Field Strength

For now everything is 1.5 T EXCEPT

Anonymized - 2181754 

I moved that one

"""

patient_data_dir = '/media/bugger/MyBook/data/leiner_data'
patient_list = os.listdir(patient_data_dir)

study_list = []
for i in patient_list:
    fs_str = ''
    print(i)
    temp_dir = os.path.join(patient_data_dir, i, 'Mri Hart')
    for sel_dir in os.listdir(temp_dir):
        data_dir = os.path.join(temp_dir, sel_dir)
        file_dir = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]

        A_dcm_list = [pydicom.read_file(x) for x in file_dir]
        for x in A_dcm_list:
            try:
                fs_str = x['MagneticFieldStrength'].value
            except KeyError:
                fs_str = ''

            if fs_str:
                print(fs_str)
                break

        if fs_str:
            break