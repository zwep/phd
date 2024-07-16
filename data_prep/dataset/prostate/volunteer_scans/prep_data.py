"""
At a certain point we have done many prostate scans..
We would like to extract all those t2 maps + b1 maps belonging to the scans
"""

import os
import reconstruction.ReadCpx as read_cpx
import helper.plot_class as hplotc
import re
import numpy as np

"""
Create necessary target directories..
"""

data_dir = '/media/bugger/MyBook/data/7T_scan/prostate'
target_dir = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection'

t2w_dir = os.path.join(target_dir, 't2w')
b1map_dir = os.path.join(target_dir, 'b1map')
body_mask_dir = os.path.join(target_dir, 'body_mask')
butt_mask_dir = os.path.join(target_dir, 'muscle_mask')
butt_fat_mask_dir = os.path.join(target_dir, 'subcutaneous_fat_mask')
prostate_mask_dir = os.path.join(target_dir, 'prostate_mask')

list_of_dir = [t2w_dir, b1map_dir, body_mask_dir, prostate_mask_dir, butt_mask_dir, butt_fat_mask_dir]
# Create directories when they are not there.
for temp_dir in list_of_dir:
    if not os.path.isdir(temp_dir):
        print('Creating ', temp_dir)
        os.mkdir(temp_dir)
    else:
        print('Already exists ', temp_dir)

# All usefull folders start with '2020' or '2021' or in thefuture.. '2022'
sub_folder = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.startswith('20')]
removing_folder = sub_folder[1]  # This should be the 2020_07_01 scan
print('Because of bad image quality, removing folder ', removing_folder)
del sub_folder[1]


"""
Load data, format it correctly, and store it.
"""

# Lastig om al die shit te onderscheiden..
for i_sub_folder in sub_folder:
    for ddir, sdir, ffiles in os.walk(i_sub_folder):
        if 'dicom' not in ddir.lower():
            if ffiles:
                # Cool nu kan ik hier alle B1 mappen en T2 gewogen shit eruit halen
                # Opslaan als Numpy in de daarvoor bestemde directory
                t2_files = [x for x in ffiles if x.endswith('.cpx') and 't2' in x]
                b1_files = [x for x in ffiles if x.endswith('.cpx') and 'b1map' in x]

                print('t2files', t2_files)
                for i_t2_file in t2_files:
                    t2_file_path = os.path.join(ddir, i_t2_file)
                    t2_file_name, _ = os.path.splitext(i_t2_file)
                    t2_target_file_path = os.path.join(t2w_dir, t2_file_name)

                    # Check if we have already created the file before..
                    if not os.path.isfile(t2_target_file_path + '.npy'):
                        print('Creating T2W file ', t2_file_name)
                        cpx_obj = read_cpx.ReadCpx(t2_file_path)
                        t2_img = cpx_obj.get_cpx_img()
                        t2_img_rot = np.rot90(np.squeeze(t2_img), axes=(-2, -1))
                        # Now save the t2 img..
                        # print('Shape of array', t2_img_rot.shape)
                        np.save(t2_target_file_path, t2_img_rot[-8:])

                # Store B1 maps, recover them using AFI..
                # Simple calculation, although I am unsure what the TR fraction is.
                print('b1 files ', b1_files)
                for i_b1_file in b1_files:
                    b1_file_path = os.path.join(ddir, i_b1_file)
                    b1_file_name, _ = os.path.splitext(i_b1_file)
                    b1_target_file_path = os.path.join(b1map_dir, b1_file_name)

                    if not os.path.isfile(b1_target_file_path + '.npy'):
                        print('Creating B1 file ', b1_file_name)
                        cpx_obj = read_cpx.ReadCpx(b1_file_path)
                        b1_img = cpx_obj.get_cpx_img()
                        b1_sum = np.squeeze(b1_img.sum(axis=0))

                        afi_r = b1_sum[:, 1] / b1_sum[:, 0]  # Dit is ook een gok
                        afi_n = 1.5  # Geen idee
                        afi = np.arccos((afi_n * afi_r - 1) / (afi_n - afi_r))
                        afi = np.rot90(afi, axes=(-2, -1))
                        sel_loc = 0  # Er zijn er drie.. ik gok ff op deze
                        afi = afi[sel_loc]

                        # Now save the t2 img..
                        np.save(b1_target_file_path, afi)


"""
Here we create full body masks for the measured 7T data

We select them manually because we want to create masks..  
"""

# Filter the files with what have already one
file_list_t2w = os.listdir(t2w_dir)
# current_body_mask = os.listdir(body_mask_dir)
current_body_mask = os.listdir(butt_mask_dir)
current_prostate_mask = os.listdir(prostate_mask_dir)
filtered_file_list_t2w_0 = [x for x in file_list_t2w if x not in current_body_mask]
filtered_file_list_t2w_1 = [x for x in file_list_t2w if x not in current_prostate_mask]
# Yeah jsut check two mask dirs to be sure
filtered_file_list_t2w = list(set(filtered_file_list_t2w_0).intersection(set(filtered_file_list_t2w_1)))

n_files = len(filtered_file_list_t2w)
print('Amount of files to mask for...', n_files)

# Select file
counter = -1


counter += 1
i_file = filtered_file_list_t2w[counter]

# Create load paths, and target paths, load files
prostate_file = os.path.join(t2w_dir, i_file)
body_mask_target_path = os.path.join(body_mask_dir, i_file)
prostate_mask_target_path = os.path.join(prostate_mask_dir, i_file)
butt_mask_target_path = os.path.join(butt_mask_dir, i_file)
butt_fat_mask_target_path = os.path.join(butt_fat_mask_dir, i_file)
print(prostate_file)
sel_array = np.load(prostate_file)

# Create full body mask
body_mask_obj = hplotc.MaskCreator(sel_array, main_title='Body mask creator')
np.save(body_mask_target_path, body_mask_obj.mask)

# Create prostate mask
prostate_mask_obj = hplotc.MaskCreator(sel_array, main_title='Prostate mask creator')
np.save(prostate_mask_target_path, prostate_mask_obj.mask)

# Create butt mask
butt_mask_obj = hplotc.MaskCreator(sel_array, main_title='Butt mask creator')
np.save(butt_mask_target_path, butt_mask_obj.mask)

# Create fat butt mask
butt_fat_mask_obj = hplotc.MaskCreator(sel_array, main_title='Butt fat mask creator')
np.save(butt_fat_mask_target_path, butt_fat_mask_obj.mask)

"""
Inspect masks that were created...
"""

body_mask_array = [np.load(os.path.join(body_mask_dir, x)) for x in os.listdir(body_mask_dir)]
hplotc.SlidingPlot(np.array(body_mask_array))

prostate_mask_array = [np.load(os.path.join(prostate_mask_dir, x)) for x in os.listdir(prostate_mask_dir)]
hplotc.SlidingPlot(np.array(prostate_mask_array))
