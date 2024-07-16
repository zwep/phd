
"""
In this file we will collect all the data and place it in the right spot...
"""


import numpy as np
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import os
import pydicom
import sigpy
import scipy.io
import helper.array_transf as harray
import reconstruction.sensitivity_map as sens_map
import matplotlib.pyplot as plt

"""

Data from Numpy Leiner

"""
ddata_rho = '/media/bugger/MyBook/data/leiner_data'
destdir_rho = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/non_registered/rho'

rho_array_list = []
sel_slice_list = []

rho_dir_counter = 0
rho_dir_file_counter = 0

img_counter = 0
img_counter_skip = len(rho_array_list)

for d, _, f in os.walk(ddata_rho):
    # This filters the directories that should contain 'survey' scans
    if 'locali' in d.lower():
        print(d, rho_dir_counter, rho_dir_file_counter)
        rho_dir_counter += 1
        if f:
            file_list = [x for x in f if x.endswith('dcm')]

            n_files = len(file_list)
            rho_dir_file_counter += n_files
            img_counter += 1

            # Load the data from the path...
            temp_array = []
            for ifile in file_list:
                file_name = os.path.join(d, ifile)
                rho_array = pydicom.dcmread(file_name).pixel_array
                temp_array.append(rho_array)

            # Filter on size so that stack works...
            filter_array = [x for x in temp_array if x.shape == (192, 192)]
            if len(filter_array):
                plot_array = np.stack(filter_array)

                # Visualize the data
                hplotc.SlidingPlot(plot_array)
                for i in range(20):
                    print(i, end='\r')
                    plt.pause(1)

                # And select the input
                sel_slice = input()
                if sel_slice.isdigit():
                    sel_slice_int = int(sel_slice)
                    sel_array = plot_array[sel_slice_int]
                    rho_array_list.append(sel_array)
                    print('Chosen slice ', sel_slice)
                    sel_slice_list.append((sel_slice_int, d))
                else:
                    print('Skipping ', d)
                    sel_slice_list.append((-1, d))
                    next

                hplotf.close_all()
            else:
                next


# I dont really care what name I give them.. just patient 0 .. 44
for i, i_array in enumerate(rho_array_list):
    patient_nr = str(i).zfill(2)
    file_name = f'leiner_patient_{patient_nr}.npy'
    file_dir = os.path.join(destdir_rho, file_name)
    np.save(file_dir, i_array)


destdir_rho = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/non_registered/rho'
temp_list_files = os.listdir(destdir_rho)

for i_file in temp_list_files:
    file_dir = os.path.join(destdir_rho, i_file)
    temp_array = np.load(file_dir)
    print(temp_array.shape)
    plt.imshow(np.abs(temp_array))
    plt.show()

""" 
        Data from Bart's simulations
"""


import numpy as np
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import os
import pydicom
import sigpy
import scipy.io
import helper.array_transf as harray
import reconstruction.sensitivity_map as sens_map
import matplotlib.pyplot as plt

ddata_b1plus = '/home/bugger/Documents/data/simulation/cardiac/b1_plus'
dest_dir_b1_plus = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/non_registered/b1plus'
list_files_b1_plus = [x for x in os.listdir(ddata_b1plus) if x.endswith('.mat')]

rho_array_list = []
sel_slice_list = []

rho_dir_counter = 0
rho_dir_file_counter = 0

img_counter = 0
img_counter_skip = len(rho_array_list)


for i_file in list_files_b1_plus:
    file_name = os.path.join(ddata_b1plus, i_file)
    load_mat_file = scipy.io.loadmat(file_name)
    output_mat = load_mat_file['Output']
    b1_plus_array = np.moveaxis(output_mat['B1p'][0][0], -1, 0)
    print('Size of B1-plus ', b1_plus_array.shape)
    depth_b1_plus = b1_plus_array.shape[-1]

    hplotc.SlidingPlot(np.moveaxis(b1_plus_array.sum(axis=0), -1, 0))
    plt.show()

    # And select the input
    sel_slice = input()
    if sel_slice.isdigit():
        sel_slice_int = int(sel_slice)
        sel_array = b1_plus_array[:, :, :, sel_slice_int]
        rho_array_list.append(sel_array)
        print('Chosen slice ', sel_slice)
        sel_slice_list.append((sel_slice_int, i_file))
    else:
        print('Skipping ', d)
        sel_slice_list.append((-1, d))
        next

    hplotf.close_all()

import re

# Now store it..
for i, i_array in enumerate(rho_array_list):
    i_file = sel_slice_list[i][1]
    v_number = re.findall("(V[0-9]+)", i_file)[0]
    dest_dir = os.path.join(dest_dir_b1_plus, f'array_{v_number}')
    print(dest_dir)
    # Now store it...
    np.save(dest_dir, i_array)

# Read it again and check the orientation...

dest_dir_b1_plus = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/non_registered/b1plus'
temp_list_files = os.listdir(dest_dir_b1_plus)

for i_file in temp_list_files:
    file_dir = os.path.join(dest_dir_b1_plus, i_file)
    temp_array = np.load(file_dir)
    print(temp_array.shape)
    plt.imshow(np.abs(temp_array).sum(axis=0))
    plt.show()

"""
        Data from my own scans on 7T
"""

# Load data..
ddata_b1_shimseries_train = '/home/bugger/Documents/data/7T/b1shimsurv_all_channels/train/input'
ddata_b1_shimseries_test = '/home/bugger/Documents/data/7T/b1shimsurv_all_channels/test/input'
ddata_b1_shimseries_validation = '/home/bugger/Documents/data/7T/b1shimsurv_all_channels/validation/input'
dest_dir_b1_minus = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/non_registered/b1minus'

list_files_b1_shimseries_validation = os.listdir(ddata_b1_shimseries_validation)
list_files_b1_shimseries_test = os.listdir(ddata_b1_shimseries_test)
list_files_b1_shimseries_train = os.listdir(ddata_b1_shimseries_train)
list_files_b1_shimseries = list_files_b1_shimseries_validation + list_files_b1_shimseries_test + list_files_b1_shimseries_train

# First doing everything in slice 14.. since that was quite ok...
for i_file in list_files_b1_shimseries:
    if i_file in list_files_b1_shimseries_validation:
        file_dir = os.path.join(ddata_b1_shimseries_validation, i_file)
    elif i_file in list_files_b1_shimseries_test:
        file_dir = os.path.join(ddata_b1_shimseries_test, i_file)
    elif i_file in list_files_b1_shimseries_train:
        file_dir = os.path.join(ddata_b1_shimseries_train, i_file)

    dest_dir = os.path.join(dest_dir_b1_minus, i_file)
    b1_array = np.load(file_dir)
    print('shape of input..', b1_array.shape)

    # Deviceplacement can be done with cupy..
    # sel_gpu = 0  # This can/should be changes
    # device_cuda = sigpy.Device(sel_gpu)
    # Is this the correct summation..?
    tx_interference = b1_array.sum(axis=1)
    # hplotf.plot_3d_list(tx_interference, augm='np.abs')
    kspace_array = harray.transform_image_to_kspace_fftn(tx_interference, dim=(-2, -1))
    # espirit_obj = sens_map.EspiritCalib(kspace_array, device=device_cuda)
    espirit_obj = sens_map.EspiritCalib(kspace_array)
    img_b1_min_array = espirit_obj.run()
    hplotf.plot_3d_list(img_b1_min_array[np.newaxis], augm='np.abs')

    # Now store it...
    np.save(dest_dir, img_b1_min_array)

# I have separated some files by hand.. lets see what is in there
filtered_dir_b1_mins = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/non_registered/b1minus/filtered'
aligned_dir_b1_mins = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/non_registered/b1minus/aligned'
temp_list_files = os.listdir(filtered_dir_b1_mins)
orientation_list = []

for i_file in temp_list_files:
    file_dir = os.path.join(filtered_dir_b1_mins, i_file)
    dest_file_dir = os.path.join(aligned_dir_b1_mins, i_file)
    temp_array = np.load(file_dir)
    plt.imshow(np.abs(temp_array).sum(axis=0))
    plt.show()
    temp_input = input('Orientation of the back')

    if temp_input == 'left':
        temp_array = temp_array[:, :, ::-1]
    
    np.save(dest_file_dir, temp_array)
    orientation_list.append(temp_input)


dest_dir_alligned = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/non_registered/b1minus/aligned'
temp_list_files = os.listdir(dest_dir_alligned)

for i_file in temp_list_files:
    file_dir = os.path.join(dest_dir_alligned, i_file)
    temp_array = np.load(file_dir)
    plt.imshow(np.abs(temp_array).sum(axis=0))
    plt.show()

