import shutil
import matplotlib.pyplot as plt
import helper.plot_class as hplotc
import h5py
import os
import numpy as np
import helper.misc as hmisc
import helper.array_transf as harray

# Now inspect data and make train/test/validation split
ddata = '/media/bugger/MyBook/data/7T_data/b1_shim_series'
dtarget = '/media/bugger/MyBook/data/dataset/b1_shim_series'

hmisc.create_datagen_dir(dtarget)
file_list_h5 = [x for x in os.listdir(ddata) if x.endswith('h5')]
file_list_h5 = [x for x in file_list_h5 if 'radial' not in x]

# I checked manually  all files below and created a filter script
for x in file_list_h5:
    print('File ', x)
    file_shim = os.path.join(ddata, x)
    A = hmisc.load_array(file_shim)
    A_shape = A.shape
    print(A_shape)
    abs_shim_array = np.abs(A[0] + 1j * A[1]).sum(axis=0).sum(axis=0)
    shim_body_mask = harray.get_treshold_label_mask(abs_shim_array)

    fig_obj = hplotc.ListPlot([shim_body_mask, abs_shim_array])
    fig_obj.figure.savefig(f'/home/bugger/{x}.png')
    hplotc.close_all()


# This was the destination name.. manually created though
dfilter_file = '/media/bugger/MyBook/data/7T_data/b1_shim_series/filter_file_names'
