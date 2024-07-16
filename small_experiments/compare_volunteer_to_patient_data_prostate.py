
"""
Alexander says that because one is SENSE reconstructed and the other Sum of Absolutes.. the characteristics are different.
I dont think that is the case...
"""

import os
import pydicom
import numpy as np
import helper.plot_class as hplotc

main_data_path = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/'
measured_path = os.path.join(main_data_path, 't2w')
body_mask_path = os.path.join(main_data_path, 'body_mask')
prostate_mask_path = os.path.join(main_data_path, 'prostate_mask')
muscle_mask_path = os.path.join(main_data_path, 'muscle_mask')
subcutaneous_fat_mask_path = os.path.join(main_data_path, 'subcutaneous_fat_mask')

body_mask_file_list = sorted([os.path.join(body_mask_path, x) for x in os.listdir(body_mask_path)])
prostate_mask_file_list = sorted([os.path.join(prostate_mask_path, x) for x in os.listdir(prostate_mask_path)])
muscle_mask_file_list = sorted([os.path.join(muscle_mask_path, x) for x in os.listdir(muscle_mask_path)])
subcutaneous_fat_mask_file_list = sorted([os.path.join(subcutaneous_fat_mask_path, x) for x in os.listdir(subcutaneous_fat_mask_path)])

file_list = sorted([os.path.join(measured_path, x) for x in os.listdir(measured_path) if x in os.listdir(subcutaneous_fat_mask_path)])

volunteer_files = []
for i in [18, 17, 16, 13, 12, 11, 6, 5, 4, 0]:
    load_file = file_list[i]
    file_name, _ = os.path.splitext(os.path.basename(load_file))
    body_mask_file = body_mask_file_list[i]
    prostate_mask_file = prostate_mask_file_list[i]
    muscle_mask_file = muscle_mask_file_list[i]
    subcutaneous_fat_mask_file = subcutaneous_fat_mask_file_list[i]

    input_cpx = np.load(load_file)
    volunteer_files.append(np.abs(input_cpx).sum(axis=0))


hplotc.SlidingPlot(np.array(volunteer_files))

# Gather patient data...