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
      
--> This file is very similar to prep_phantom_dataset!!

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


ddata = '/nfs/arch11/researchData/USER/emeliado/PINN_FDTD_with_Seb/MatLab_Dataset_PINN_FDTD_Duke&Ella'
ddest = '/home/sharreve/local_scratch/mri_data/pinn_fdtd_dukella'
file_list = os.listdir(ddata)
file_list = [x for x in file_list if x.endswith('mat')]
"""
Ella_0 (Array Radius 177.0mm - Dipole Angle 31.1deg).mat                               
Ella_0 (Array Radius 179.0mm - Dipole Angle 59.2deg).mat                               
Ella_0 (Array Radius 170.0mm - Dipole Angle 55.3deg).mat                               
Duke_0 (Array Radius 168.0mm - Dipole Angle 236.6deg).mat                              
Ella_0 (Array Radius 154.0mm - Dipole Angle 258.2deg).mat                              
Duke_0 (Array Radius 193.0mm - Dipole Angle 254.2deg).mat                              
Duke_0 (Array Radius 150.0mm - Dipole Angle 305.8deg).mat                              
Duke_0 (Array Radius 188.0mm - Dipole Angle 338.3deg).mat                              
Duke_0 (Array Radius 165.0mm - Dipole Angle 138.6deg).mat                              
Duke_0 (Array Radius 174.0mm - Dipole Angle 13.3deg).mat                               
Ella_0 (Array Radius 198.0mm - Dipole Angle 127.2deg).mat                              
Duke_0 (Array Radius 167.0mm - Dipole Angle 260.0deg).mat                              
Ella_0 (Array Radius 175.0mm - Dipole Angle 58.7deg).mat                               
Ella_0 (Array Radius 167.0mm - Dipole Angle 317.7deg).mat                              
Ella_0 (Array Radius 165.0mm - Dipole Angle 21.0deg).mat                               
Duke_0 (Array Radius 181.0mm - Dipole Angle 157.8deg).mat                              
Duke_0 (Array Radius 154.0mm - Dipole Angle 149.9deg).mat                              
Ella_0 (Array Radius 168.0mm - Dipole Angle 315.1deg).mat      
"""

# Train on Ella.. test on Duke...?
ella_files = [x for x in file_list if 'Ella' in x]
duke_files = [x for x in file_list if 'Duke' in x]
n_duke_files = len(duke_files)
train_phantoms = ella_files
val_phantoms = duke_files[:n_duke_files//2]
test_phantoms = duke_files[n_duke_files//2:]

print('Example of test files ', test_phantoms)

# Create the target directories...
hmisc.create_datagen_dir(ddest,  type_list=('test', 'validation', 'train'), data_list=('input', 'input_sigma', 'input_eps', 'target_B', 'target_E'))

for i_file in file_list:
    print("Processing file ", i_file)
    file_path = os.path.join(ddata, i_file)
    file_name = re.sub('\.mat', '', i_file)
    # file_name = re.sub(' ', '', i_file)
    mat_obj = hmisc.load_array(file_path)
    if i_file in train_phantoms:
        split_type = 'train'
    elif i_file in test_phantoms:
        split_type = 'test'
    else:
        split_type = 'validation'
    print("Put into split: ", split_type)
    input_dict, target_dict = get_maxwell_components(mat_obj)
    store_array(input_dict, target_dict, ddir=ddest, dsplit=split_type, file_name=file_name)