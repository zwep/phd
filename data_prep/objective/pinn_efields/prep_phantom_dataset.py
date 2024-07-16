import h5py
import numpy as np
import helper.plot_class as hplotc
import helper.misc as hmisc
import helper.array_transf as harray
import os
import re

"""
Get number of files
Split in train/test/validation
Per file:
  Find the correct split
      Extract the B- and E-fields. Reshape them, store them as.... h5 files. Dont normalize
      Extract the rho/eps/sigma fields. Reshape them, dont normalize
          --> What about using a dimensionless form of the maxwell equation.. Does that help?
      Store them in de desired folder/location
"""


def get_maxwell_components(mat_obj):
    reshape_size = tuple(mat_obj['Grid_size'][0][::-1])
    # Define the 'constant' arrays...
    rho_array = mat_obj['rho'].reshape(reshape_size)
    # Conductivity sigma
    sigma_array = mat_obj['sigma'].reshape(reshape_size)
    # Permitivity epsilon
    eps_array = mat_obj['eps'].reshape(reshape_size)
    # Permeability mu...?
    # Reshape the B and E fields
    B_array = mat_obj['Bfield'].reshape(reshape_size + (3,))
    E_array = mat_obj['Efield'].reshape(reshape_size + (3,))
    return {"rho": rho_array, "sigma": sigma_array, "eps": eps_array}, {"B": B_array, "E": E_array}


def store_array(input_dict, target_dict, ddir, dsplit, file_name):
    # dict_object contains rho/sigma/eps
    # ddir is the main directory
    # dsplit contians train/test/validation
    for k, v in input_dict.items():
        if 'rho' in k:
            # This is needed so that we can createa a file_list later on with the data generator
            ddest = os.path.join(ddir, dsplit, "input", file_name + '.h5')
        else:
            ddest = os.path.join(ddir, dsplit, "input_" + k, file_name + '.h5')
        print(k, v.shape)
        # Nog even nadenken over wat voor type data ik wil..
        with h5py.File(ddest, 'w') as f:
            f.create_dataset('data', data=v)
    for k, v in target_dict.items():
        ddest = os.path.join(ddir, dsplit, "target_" + k, file_name + '.h5')
        print(k, v.shape)
        # Nog even nadenken over wat voor type data ik wil..
        with h5py.File(ddest, 'w') as f:
            f.create_dataset('data', data=v)


ddata = '/nfs/arch11/researchData/USER/emeliado/PINN_FDTD_with_Seb/MatLab_Dataset_PINN_FDTD'
ddest = '/home/sharreve/local_scratch/mri_data/pinn_fdtd'
file_list = os.listdir(ddata)
file_list = [x for x in file_list if x.endswith('mat')]


re_phantom = re.compile('Phantom_([0-9]+)')
phantom_names = list(set(sorted([int(re_phantom.findall(x)[0]) for x in file_list if re_phantom.findall(x)])))
n_phantoms = len(phantom_names)
n_train = int(0.70 * n_phantoms)
n_val = int(0.2 * n_phantoms)
n_test = int(0.1 * n_phantoms)
train_phantoms = phantom_names[:n_train]
val_phantoms = phantom_names[n_train:(n_train + n_val)]
test_phantoms = phantom_names[(n_train + n_val):]

# Create the target directories...
hmisc.create_datagen_dir(ddest,  type_list=('test', 'validation', 'train'), data_list=('input_rho', 'input_sigma', 'input_eps', 'target_B', 'target_E'))

for i_file in file_list:
    print("Processing file ", i_file)
    file_path = os.path.join(ddata, i_file)
    file_name = hmisc.get_base_name(i_file)
    mat_obj = hmisc.load_array(file_path)
    phantom_number = int(re_phantom.findall(i_file)[0])
    if phantom_number in train_phantoms:
        split_type = 'train'
    elif phantom_number in test_phantoms:
        split_type = 'test'
    else:
        split_type = 'validation'
    print("Put into split: ", split_type)
    input_dict, target_dict = get_maxwell_components(mat_obj)
    store_array(input_dict, target_dict, ddir=ddest, dsplit=split_type, file_name=file_name)
