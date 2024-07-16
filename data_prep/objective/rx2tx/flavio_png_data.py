# encoding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc

"""
Extract the images from flavios dataset.. see if we can find a mapping between them.
"""

from PIL import Image


dir_data = '/media/bugger/MyBook/data/simulation/flavio/Dataset_Seb'
dest_dir = '/media/bugger/MyBook/data/simulation/flavio/Dataset_Seb_selected'

hmisc.create_datagen_dir(dest_dir)

list_files = os.listdir(dir_data)
# Extract the sequence of this..
# re.sub(...)(0-9){2}.png
# Groepeer deze in train/test split
# Dan input/target switchen//
import torch.nn


for i_file in os.listdir(dir_data):
    read_file = os.path.join(dir_data, i_file)
    file_name, ext = os.path.splitext(i_file)
    input_file = file_name
    target_file = os.path.join(dest_dir, i_file)


A = np.array(Image.open(read_file))
img_x = img_y = 256
ny = 2
nx = 5
extracted_input = []
extracted_target = []
for ix in range(nx):
    for iy in range(ny):
        temp = A[(ix * img_x):( (ix + 1) * img_x), (iy * img_y):( (iy  + 1) * img_y)]
        if iy == 0:
            extracted_input.append(temp)
        else:
            extracted_target.append(temp)

extracted_input = [np.moveaxis(x, -1, 0) for x in extracted_input]
extracted_target = [np.moveaxis(x, -1, 0) for x in extracted_target]

input_complex = [real + 1j * imag for mask, real, imag in extracted_input]
target_complex = [real + 1j * imag for mask, real, imag in extracted_target]




hplotf.plot_3d_list(extracted_list)
augm_list = ['np.abs', 'np.real', 'np.imag', 'np.angle']
for augm in augm_list:
    hplotf.plot_3d_list(extracted_complex, augm=augm)
