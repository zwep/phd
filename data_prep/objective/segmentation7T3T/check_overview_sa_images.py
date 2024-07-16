"""
Here we are going to visually inspect the unfolded images
"""

import os
import nrrd
import numpy as np
import helper.array_transf as harray
import helper.plot_class as hplotc

ddat = '/media/bugger/MyBook/data/7T_data/unfolded_sa'
d_files = [os.path.join(ddat, x) for x in os.listdir(ddat) if x.endswith('nrrd')]
counter = -1


hplotc.close_all()
counter += 1
sel_file = d_files[counter]
print('Counter ', counter)
print(sel_file)
loaded_array, _ = nrrd.read(sel_file)
loaded_array_cpx = harray.to_complex(loaded_array)
print('Shape: ')
print(', '.join(list([str(x) for x in loaded_array_cpx.shape])))
hplotc.SlidingPlot(loaded_array_cpx)


import scipy.ndimage
abs_img = np.abs(loaded_array_cpx[0][0])
angle_range = np.linspace(0, 270, 10)
res = [scipy.ndimage.rotate(abs_img, angle=x) for x in angle_range]
hplotc.ListPlot([res], subtitle=[list(angle_range)])

hplotc.MaskCreator(res[-4][None])

"""
Read in the data that I uploaded...
"""

import nrrd


def read_complex_nrrd(file_path):
    data_array, data_header = nrrd.read(file_path)
    data_real = np.take(data_array, 0, axis=-1)
    data_imag = np.take(data_array, 1, axis=-1)
    data_cpx = data_real + 1j * data_imag
    return data_cpx

counter = 0
total_loc = 0
ddata = '/media/bugger/MyBook/data/7T_data/unfolded_sa_filtered'
for i_file in os.listdir(ddata):
    single_file = os.path.join(ddata, i_file)
    if single_file.endswith('nrrd'):
        # loaded_array, _ = nrrd.read(single_file)
        # loaded_array_cpx = harray.to_complex(loaded_array)
        loaded_array_cpx = read_complex_nrrd(single_file)
        n_loc = loaded_array_cpx.shape[0]
        print(n_loc)
        total_loc += n_loc
        n_card = loaded_array_cpx.shape[1]
        hplotc.SlidingPlot(loaded_array_cpx)
        counter += 1

    if counter > 5:
        break

# Loop over the original ones again...
ddata = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac'
counter = 0
for d, _, f in os.walk(ddata):
    filter_f = [x for x in f if 'sa' in x and x.endswith('npy') and 'ppu' in x]
    if len(filter_f):
        counter += 1
        for sel_file in filter_f:
            file_path = os.path.join(d, sel_file)
            A = np.load(file_path)
            hplotc.SlidingPlot(A)
    if counter > 10:
        break

