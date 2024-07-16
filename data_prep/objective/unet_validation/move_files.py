# encoding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc

"""

"""

# Since everything is
train_id = 'validate'
cur_dir = f'/home/bugger/Documents/data/grand_challenge/data/{train_id}'
tgt_dir = f'/home/bugger/Documents/data/grand_challenge/data/{train_id}/target'
inp_dir = f'/home/bugger/Documents/data/grand_challenge/data/{train_id}/input'

file_list = os.listdir(cur_dir)
mask_file = [x for x in file_list if 'mask' in x]
image_file = [x for x in file_list if 'image' in x]

# Move mask to tgt dir
for ifile in mask_file:
    orig_file = os.path.join(cur_dir, ifile)
    dst_file = os.path.join(tgt_dir, ifile)
    os.rename(orig_file, dst_file)

# Move image to input dir
for ifile in image_file:
    orig_file = os.path.join(cur_dir, ifile)
    dst_file = os.path.join(inp_dir, ifile)
    os.rename(orig_file, dst_file)
