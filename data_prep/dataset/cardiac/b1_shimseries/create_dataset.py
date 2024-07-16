import shutil
import h5py
import os
import numpy as np
import helper.misc as hmisc
import helper.array_transf as harray

# Now inspect data and make train/test/validation split
ddata = '/media/bugger/MyBook/data/7T_data/b1_shim_series'
dtarget = '/media/bugger/MyBook/data/dataset/b1_shim_series'

# This allowed met to filter based on quality...
# It is manually created with a commented script below. Yay
dfilter_file = '/media/bugger/MyBook/data/7T_data/b1_shim_series/filter_file_names'
with open(dfilter_file, 'r') as f:
    filter_file_name_list = [x.strip() for x in f.readlines()]

hmisc.create_datagen_dir(dtarget)
file_list_h5 = [x for x in os.listdir(ddata) if x.endswith('h5')]
file_list_h5 = [x for x in file_list_h5 if 'radial' not in x]
# Filter based on the files that were of bad quality
filter_file_list_h5 = [x for x in file_list_h5 if not (hmisc.get_base_name(x) in filter_file_name_list)]

# Create train-test-val split
n_files = len(file_list_h5)
perc_train = 0.8
perc_validation = 0.1
perc_test = 0.1
n_train = int(n_files * perc_train)
n_test = int(n_files * perc_test)
n_validation = int(n_files * perc_validation)

train_files = file_list_h5[:n_train]
test_files = file_list_h5[n_train:(n_train + n_test)]
validation_files = file_list_h5[-n_validation:]


def copy_data_files(file_list, origin, target):
    for i_file in file_list:
        print('Copying ', i_file)
        origin_file = os.path.join(origin, i_file)
        target_file = os.path.join(target, i_file)
        dirname_target = os.path.dirname(target)
        dmask = os.path.join(dirname_target, 'mask')

        if not os.path.isdir(dmask):
            os.makedirs(dmask)

        # Create a mask on the fly
        A = hmisc.load_array(origin_file)
        abs_shim_array = np.abs(A[0] + 1j * A[1]).sum(axis=0).sum(axis=0)
        shim_body_mask = harray.get_treshold_label_mask(abs_shim_array)
        shutil.copy(origin_file, target_file)
        # Store the mask as an h5 file
        with h5py.File(os.path.join(dmask, i_file), 'w') as f:
            f.create_dataset('data', data=shim_body_mask)


copy_data_files(train_files, origin=ddata, target=os.path.join(dtarget, 'train/input'))
copy_data_files(test_files, origin=ddata, target=os.path.join(dtarget, 'test/input'))
copy_data_files(validation_files, origin=ddata, target=os.path.join(dtarget, 'validation/input'))

