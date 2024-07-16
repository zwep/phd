import os
import collections
import re
import h5py
import numpy as np
import helper.misc as hmisc

"""
So...

the numbers of slice I report in the paper are 46 440

This comes from combining 23 body models and 40 patient images with 90 slices each.
The important part here is that FIRST each body model/patient is put into its specific train/test/val split
THEN the images are created.

This makes for the following distribution:

train: 17 body models, 28 patients
test: 4 body models, 8 patients 
val: 2 body models, 4 patients
 
Now... with 90 slices per subject this becomes:

90 * (17 * 28 + 4 * 8 + 2 * 4) = 46 440

which is NOT the same as 40*23*90
 
"""

ddata_h5 = '/local_scratch/sharreve/mri_data/registrated_h5'
subdir_list = ['train', 'test', 'validation']
slice_list = []
name_list = []
for i_sub in subdir_list:
    sub_slice_list = []
    sub_name_list = []
    file_dir = os.path.join(ddata_h5, i_sub, 'input')
    file_list = os.listdir(file_dir)
    for i_file in file_list:
        sel_file = os.path.join(file_dir, i_file)
        re_obj = re.findall('(M[0-9]+)', i_file)
        if re_obj:
            m_name = re_obj[0]
        else:
            m_name = 'Duke'
        with h5py.File(sel_file, 'r') as f:
            n_slice = f['data'].shape[0]
        sub_slice_list.append(n_slice)
        sub_name_list.append(m_name)
    slice_list.append(sub_slice_list)
    name_list.append(sub_name_list)


for i in slice_list:
    print(sum(i), np.mean(i), len(i))

for i in name_list:
    col_obj = collections.Counter(i)
    print('Number of M numbers', len(col_obj))
    print(col_obj)



# Checking the smoothing with uneven and even kernel sizes
import helper.array_transf as harray
import numpy as np
A = np.random.rand(10, 10)
A_even = harray.smooth_image(A, n_kernel=8)
A_uneven = harray.smooth_image(A, n_kernel=9)
import helper.plot_class as hplotc
hplotc.ListPlot([A_even, A_uneven])