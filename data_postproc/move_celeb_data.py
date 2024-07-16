import shutil
import os
import helper.misc as hmisc

"""
Here we are going to move all the celeb files in one folder to a train/test/validation split for data generator purposes

"""

orig_dir = '/home/bugger/Documents/data/celeba/img_align_celeba/img_align_celeba'
dest_dir = '/home/bugger/Documents/data/celeba'

type_list = ['train', 'test', 'validation']
data_list = ['input', 'target']
file_list = os.listdir(orig_dir)
n_files = len(file_list)

information_container = hmisc.create_datagen_dir(dest_dir, type_list=type_list, data_list=data_list)

# Do the split
if len(type_list) == 2:
    if 'test' in type_list and 'train' in type_list:
        print('We split the data into train/test (70/30)')
        n_train = int(0.70 * n_files)
        train_files = file_list[:n_train]
        validation_files = []
        test_files = file_list[n_train:]
    else:
        print('Unkown content of type list ', type_list)
elif len(type_list) == 3:
    if 'test' in type_list and 'train' in type_list and 'validation' in type_list:
        print('We split the data into train/validation/test (65/10/25)')
        n_train = int(0.65 * n_files)
        n_validate = int(0.10 * n_files)
        train_files = file_list[:n_train]
        validation_files = file_list[n_train:(n_train + n_validate)]
        test_files = file_list[(n_train + n_validate):]
    else:
        print('Unkown content of type list ', type_list)
else:
    print('Unkown content of type list ', type_list)

dest_train = information_container['train']['input']
[shutil.move(os.path.join(orig_dir, x), dest_train) for x in train_files]

dest_test = information_container['test']['input']
[shutil.move(os.path.join(orig_dir, x), dest_test) for x in test_files]

dest_validation = information_container.get('validation', None)
if dest_validation is not None:
    dest_validation = dest_validation['input']
    [shutil.move(os.path.join(orig_dir, x), dest_validation) for x in validation_files]
