

"""
Here we aer going to move the Rx/Tx data to the new directory
"""

import getpass
import sys
import os
import numpy as np
import re
import nrrd
import json
import skimage.transform as sktransf

# Deciding which OS is being used
local_system = False
if getpass.getuser() == 'bugger':
    local_system = True

if local_system:
    project_path = "/"
    rxtx_dir = '/home/bugger/Documents/data/semireal/prostate_simulation_rxtx'
    dest_dir = '/home/bugger/Documents/data/semireal/prostate_simulation_t1t2_rxtx'

    prostate_img_dir = '/home/bugger/Documents/data/prostateimagedatabase'

else:
    project_path = "/data/seb/code/pytorch_in_mri"
    rxtx_dir = '/data/seb/semireal/prostate_simulation_rxtx'
    dest_dir = '/data/seb/semireal/prostate_simulation_t1t2_rxtx'

    prostate_img_dir = '/data/seb/prostatemriimagedatabase'

dir_result = os.path.join(prostate_img_dir, 'results/results.json')
sys.path.append(project_path)

import helper.misc as hmisc

hmisc.create_datagen_dir(dest_dir, type_list=['test', 'train', 'validation'],
                         data_list=['input', 'target', 'target_clean', 'target_t1', 'target_t2'])

with open(dir_result, 'r') as f:
    json_lines = f.readlines()

result_dict_list = [json.loads(x) for x in json_lines]
patient_id_list = [x['patient_id'] for x in result_dict_list]
target_shape = (256, 256)

# Hier onder worden input, target en target_clean vervuld...
# Walk over this..
for i_type_dir in ['test', 'train', 'validation']:
    rxtx_type_dir = os.path.join(rxtx_dir, i_type_dir)
    print(rxtx_type_dir)

    data_dir = ['input', 'target', 'target_clean']
    for i_data_dir in data_dir:
        rxtx_type_data_dir = os.path.join(rxtx_type_dir, i_data_dir)
        rxtx_files = os.listdir(rxtx_type_data_dir)
        print(f'Subdir {i_data_dir}')
        print(f'\tAmount of files {len(rxtx_files)}')

        data_dest_dir = os.path.join(dest_dir, i_type_dir, i_data_dir)

        for i_patient_id in patient_id_list:
            sel_result_dict = hmisc.filter_dict_list(result_dict_list, x_key='patient_id', sel_value=i_patient_id)

            if sel_result_dict == -1:
                # print('\t\t selected result dict came down to an error', patient_id)
                continue
            elif sel_result_dict == 999:
                # print('\t\t selected patient id had no results', patient_id)
                continue
            else:
                pass

            if i_data_dir in ['input', 'target']:

                slice_nr_rho = sel_result_dict['slice_rho']
                findall_comp = re.compile(".*" + i_patient_id + ".*__" + str(slice_nr_rho))
                filtered_rxtx_files = [x for x in rxtx_files if findall_comp.findall(x)]

                # Parse the RxTx npy data
                for i_rxtx_file in filtered_rxtx_files:
                    print(f'\t\t Working with file {i_rxtx_file}')
                    print('\t\t selected patient id has result', i_patient_id)
                    print(f'\t\t We have the correct slice nr {slice_nr_rho}')

                    npy_filename = os.path.join(rxtx_type_data_dir, i_rxtx_file)
                    target_npy_filename = os.path.join(data_dest_dir, i_rxtx_file)

                    co_y = sel_result_dict['coordinates0']
                    co_x = sel_result_dict['coordinates1']
                    print('Coordinates of subsampled')
                    print('\t', co_y, co_x)
                    rxtx_array = np.load(npy_filename)[0]
                    print('\t Shape of loaded data ', rxtx_array.shape)
                    sel_rxtx_array = rxtx_array[:, co_y[0]:co_y[1], co_x[0]:co_x[1]]

                    reszied_rxtx_array_real = np.stack([sktransf.resize(x.real, target_shape) for x in sel_rxtx_array], axis=0)
                    reszied_rxtx_array_imag = np.stack([sktransf.resize(x.imag, target_shape) for x in sel_rxtx_array], axis=0)
                    reszied_rxtx_array = reszied_rxtx_array_real + 1j * reszied_rxtx_array_imag
                    print('\t Shape of loaded data ', sel_rxtx_array.shape)
                    np.save(target_npy_filename, reszied_rxtx_array)

            if i_data_dir == 'target_clean':
                slice_nr_rho = sel_result_dict['slice_rho']
                filtered_rxtx_files = [x for x in rxtx_files if i_patient_id in x]

                for i_rxtx_file in filtered_rxtx_files:
                    print(f'\t Working with file {i_rxtx_file}')
                    print('\t selected patient id has result', i_patient_id)
                    print(f'\t Took slice nr {slice_nr_rho}')

                    # These are needed when dealing with target_clean data..
                    nrrd_filename = os.path.join(rxtx_type_data_dir, i_rxtx_file)
                    target_nrrd_filename = os.path.join(data_dest_dir, i_rxtx_file)
                    rho_array, _ = nrrd.read(nrrd_filename)
                    print('\t Shape of loaded data ', rho_array.shape)
                    sel_slice = sel_result_dict['slice_rho']
                    sel_rho_array = rho_array[:, :, sel_slice]
                    print('\t Shape of sel slice loaded data ', sel_slice)
                    print('\t Shape of sel slice loaded data ', sel_rho_array.shape)
                    np.save(target_nrrd_filename, sel_rho_array)


"""
Below we are going to do the same with the T1/T2 data...
"""

# Based this on my inspection..
train_patient_id = ['000003', '000004', '000005', '000006', '000008', '000010', '000011', '000012', '000014', '000016', '000017', '000019',
'000021', '000024']
validation_patient_id = ['000029']
test_patient_id = ['000031', '000035', '000036', '000039', '000041']
type_T12 = ['T1_series', 'T2_series']

for i_patient_id in patient_id_list:
    sel_result_dict = hmisc.filter_dict_list(result_dict_list, x_key='patient_id', sel_value=i_patient_id)

    if sel_result_dict == -1:
        # print('\t\t selected result dict came down to an error', patient_id)
        continue
    elif sel_result_dict == 999:
        # print('\t\t selected patient id had no results', patient_id)
        continue
    else:
        pass

    if i_patient_id in test_patient_id:
        data_dir = 'test'
    elif i_patient_id in train_patient_id:
        data_dir = 'train'
    elif i_patient_id in validation_patient_id:
        data_dir = 'validation'
    else:
        data_dir = ''

    for i_T12 in type_T12:

        if i_T12 == 'T1_series':
            dest_dir_t12 = os.path.join(dest_dir, data_dir, 'target_t1')
        else:
            dest_dir_t12 = os.path.join(dest_dir, data_dir, 'target_t2')

        prostate_t1t2 = os.path.join(prostate_img_dir, i_T12)
        t1t2_files = os.listdir(prostate_t1t2)
        filtered_t1t2_files = [x for x in t1t2_files if i_patient_id in x]
        print('Filtered input list ', filtered_t1t2_files)
        print('\t ', dest_dir_t12)

        for i_file in filtered_t1t2_files:
            print('Dealing with file ', i_file)
            i_file_name = os.path.splitext(i_file)[0]
            i_file_name = re.sub('../../../objective/inhomog_removal', '_', i_file_name)
            dest_file_t12 = os.path.join(dest_dir_t12, i_file_name)
            t1t2_file = os.path.join(prostate_t1t2, i_file)
            sel_t1t2_slice = sel_result_dict['slice_t1t2']
            t1t2_array, _ = nrrd.read(t1t2_file)
            sel_t1t2_array = t1t2_array[:, :, sel_t1t2_slice]

            np.save(dest_file_t12, sel_t1t2_array)