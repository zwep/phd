# encoding: utf-8

# Libraries
import re
import time
import numpy as np
import os
import nibabel as nib
import importlib
import torch
import getpass
import sys

# Determine local/remote
# Deciding which OS is being used
if getpass.getuser() == 'bugger':
    local_system = True
    manual_mode = True
else:
    import matplotlib as mpl
    mpl.use('Agg')  # Hopefully this makes sure that we can plot/save stuff
    local_system = False
    manual_mode = False

if local_system:
    ddata = '/home/bugger/Documents/data/grand_challenge/data'
    project_path = "/home/bugger/PycharmProjects/pytorch_in_mri"
else:
    ddata = '/data/seb/grand_challenge/data/final_test'
    project_path = "/home/seb/code/pytorch_in_mri"

print('Adding to path: ', project_path)
sys.path.append(project_path)

import helper.misc as hmisc
import data_generator.UnetValidation as data_gen
import objective.unet_validation.executor_unet_validation as executor


if local_system:
    dir_path = '/home/bugger/Documents/model_run/config_unet_temp'
    dir_path = '/home/bugger/Documents/model_run/unet_model'
    config_param = hmisc.convert_remote2local_dict(dir_path, path_prefix='/home/bugger/Documents/data/grand_challenge')
    dest_dir = '/home/bugger/data/grand_challenge/data/final_test/prediction'
    A = executor.ExecutorUnetValidation(config_file=config_param, debug=True)
else:
    # dest_dir = '/home/seb/data/grand_challenge/data/final_test/prediction'
    dest_dir = '/home/seb/data/prediction_unet_val'
    # config_path = '/data/seb/model_run/unet_validation_20200305_new/config_00'
    # config_path = '/data/seb/model_run/unet_validation_20200421_first_iteration/config_00'
    # config_path = '/data/seb/model_run/unet_validation_20200421_second_iteration/config_00'
    # config_path = '/data/seb/model_run/unet_validation_20200421_third_iteration/config_00'

    # A = executor.ExecutorUnetValidation(model_path=config_path, debug=True)
    import model.UNet3D
    import os
    import numpy as np
    import torch
    import nibabel as nib
    import helper.nvidia_parser as hnvidia
    import subprocess
    import json

    dir_config = '/data/seb/model_run/unet_validation_20200610/config_00/config_param.json'
    with open(dir_config, 'r') as f:
        config_param = json.loads(f.read())

    A = model.UNet3D.Unet3D(**config_param['model']['config_unet3d'])


    # dir_state_dict = '/home/bugger/Documents/data/grand_challenge/model_weights/model_weights.pt'
    # dir_state_dict = '/data/seb/model_run/unet_validation_20200421_third_iteration/config_00/model_weights.pt'
    # dir_state_dict = '/data/seb/model_run/unet_validation_20200514/config_00/model_weights.pt'
    dir_state_dict = '/data/seb/model_run/unet_validation_20200610/config_00/model_weights.pt'
    # dir_state_dict = '/data/seb/model_run/unet_validation_20200506/config_00/model_weights.pt'
    index_gpu, p_gpu = hnvidia.get_free_gpu_id(claim_memory=0.9)
    if index_gpu is not None:
        print('Status GPU')
        nvidia_cmd = ["nvidia-smi", "-q", "-d", "MEMORY", "-i"]
        cmd = nvidia_cmd + [str(index_gpu)]
        output = subprocess.check_output(cmd).decode("utf-8")
        print(output)

    device = torch.device("cuda:{}".format(index_gpu) if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(dir_state_dict, map_location=torch.device(device))
    A.load_state_dict(state_dict)
    A.to(device)

# A.config_param['dir']['doutput'] = config_path
# A.load_weights()

# Load data..
importlib.reload(data_gen)
input_shape = (256, 256, 180)
w, h, t = input_shape
# Size of output patched image
w_o = 164
t_o = 92
data_gen_unet = data_gen.UnetValidation(ddata=ddata, input_shape=input_shape, input_is_output=True,
                                        dataset_type='test', debug=True, shuffle=False)

file_list = data_gen_unet.file_list
input_dir = data_gen_unet.input_dir

for i_file in file_list[0:1]:
    # i_file = file_list[0]
    file = os.path.join(input_dir, i_file)
    file_name, file_ext = os.path.splitext(i_file)
    file_name_name, file_ext_ext = os.path.splitext(file_name)
    dest_file = os.path.join(dest_dir, file_name_name)
    dest_file = re.sub('_image', '_prediction', dest_file)

    print(f'Storing data to location {dest_file}')

    input_data = nib.load(file).get_fdata()
    print(f'Original size input data {input_data.shape}')
    W_unpad, H_unpad, T_unpad = input_data.shape
    # Pad it...
    pad_xy_axis = ((w - w_o) // 2, (w - w_o) // 2)
    pad_t_axis = ((t - t_o) // 2, (t - t_o) // 2)
    input_data = np.pad(input_data, (pad_xy_axis, pad_xy_axis, pad_t_axis))

    print(f'Padded size input data {input_data.shape}')
    # Original data size
    W, H, T = input_data.shape

    # Step size of the patched input
    # Reduced by w_o to account for patched output
    delta_x = w_o // 3  # // 2  # (w - w_o) // 4
    delta_t = t_o // 3  # // 2  # (t - t_o) // 4

    # Amount of steps for patched input
    n_x = int(np.ceil((W-w)/delta_x))
    n_t = int(np.ceil((T-t)/delta_t))

    # Size of target image
    target_shape = (W, H, T)
    target_data = np.ones(target_shape)
    scale_data = np.zeros(target_shape)

    print(f'Starting patching procedure: n_t {n_t}, n_x {n_x}')
    # Now still need to pad it with zeros...

    tens_data = torch.as_tensor(input_data[np.newaxis, np.newaxis]).float().to(device)
    tens_data = (tens_data - tens_data.min()) / (tens_data.max() - tens_data.min())

    t0 = time.time()
    for i_t in range(n_t):
        print(f'time {i_t}')
        t1 = time.time()
        print(f'Amount of seconds... {t1 - t0}')
        for i_x in range(n_x):
            for i_y in range(n_x):
                x_min = delta_x * i_x
                x_max = delta_x * i_x + w
                if i_x == n_x-1:
                    x_min = W - w
                    x_max = W

                y_min = delta_x * i_y
                y_max = delta_x * i_y + w
                if i_y == n_x-1:
                    y_min = W - w
                    y_max = W

                t_min = delta_t * i_t
                t_max = delta_t * i_t + t
                if i_t == n_t-1:
                    t_min = T - t
                    t_max = T

                x_min_o = x_min
                x_max_o = x_min_o + w_o

                y_min_o = y_min
                y_max_o = y_min_o + w_o

                t_min_o = t_min
                t_max_o = t_min_o + t_o

                # print(x_min_o, x_max_o, y_min_o, y_max_o, t_min_o, t_max_o, '- - -', x_min, f'{x_max}/{W}', y_min, f'{y_max}/{H}', t_min, f'{t_max}/{T}', '\n')

                # print('Calculating')
                with torch.no_grad():
                    temp_output = A(tens_data[:, :, y_min:y_max, x_min:x_max, t_min:t_max])

                t3 = time.time()
                # print(f'Amount of seconds... {t3 - t0}')
                # print(f'Output size.. {temp_output.shape}')

                # temp_output = derp_data[:, y_min_o:y_max_o, x_min_o:x_max_o, t_min_o:t_max_o]
                # print(temp_output.shape)

                if device.type == 'cpu':
                    temp_output = temp_output.numpy()[0][0]  # Select channel and batch
                else:
                    temp_output = temp_output.cpu().numpy()[0][0]  # Select channel and batch

                target_data[y_min_o:y_max_o, x_min_o:x_max_o, t_min_o:t_max_o] += temp_output
                scale_data[y_min_o:y_max_o, x_min_o:x_max_o, t_min_o:t_max_o] += np.ones(temp_output.shape[1:])

                del temp_output
                torch.cuda.empty_cache()

    target_data = target_data / (scale_data + 1)
    target_data = target_data[:W_unpad, :H_unpad, :T_unpad]
    target_data = target_data > (0.8 * np.max(target_data))
    target_data = target_data.astype(int)
#     target_data = 1/(1 + np.exp(-target_data))
    # import helper.plot_class as hplotc
    # hplotc.SlidingPlot(target_data[0]>0.4)
    C = nib.Nifti1Image(target_data, np.eye(4))
    nib.save(C, dest_file + '.nii.gz')
    print('Stuff is really saved')


load_local_data = False
if load_local_data:
    # Load the local data...
    import numpy as np
    import nibabel as nib
    import os
    import helper.plot_class as hplotc
    import matplotlib.pyplot as plt
    import time

    if local_system:
        # data_dir = '/home/bugger/Documents/data/grand_challenge/prediction'
        # data_dir = '/home/bugger/Documents/data/grand_challenge/prediction/prediction'
        # data_dir = '/home/bugger/Documents/data/grand_challenge/prediction/prediction_may'
        data_dir = '/home/bugger/Documents/data/grand_challenge/prediction/prediction_mid_may'
        dest_dir = '/home/bugger/Documents/data/grand_challenge/prediction/prediction_sigmoid'
        data_dir = '/home/bugger/Documents/data/grand_challenge'
    else:
        data_dir = '/home/seb/data/grand_challenge/data/final_test/prediction_may'
        dest_dir = '/home/seb/data/grand_challenge/data/final_test/prediction_softmax'

    file_list = [x for x in os.listdir(data_dir) if x.endswith('nii.gz')]
    t0 = time.time()
    for i_file in file_list:
        file = os.path.join(data_dir, i_file)
        dest_path = os.path.join(dest_dir, i_file)
        t1 = time.time()
        print(dest_path, t1 - t0)
        # if os.path.isfile(dest_path):
        #     print('Already done with ', dest_path)
        #     continue
        A = nib.load(file).get_fdata()
        hplotc.SlidingPlot(np.moveaxis(A, -1, 0))
        A = A > 0.8 * np.max(A)

        A = A.astype(int)
        hplotc.SlidingPlot(A > 0.7 * np.max(A))
        hplotc.SlidingPlot(np.moveaxis(A, -1, 0))
        A = nib.Nifti1Image(A, np.eye(4))
        nib.save(A, dest_path)


    # Inspect the data ovf softmax
    import numpy as np
    import os
    import nibabel as nib

    # dir_data = '/home/bugger/Documents/data/grand_challenge/prediction'
    dir_data = '/home/seb/data/grand_challenge/data/final_test/prediction'
    file_list = [x for x in os.listdir(dir_data) if 'old' not in x]
    i_file = file_list[1]
    file = os.path.join(dir_data, i_file)
    A = nib.load(file).get_fdata()
    hplotc.SlidingPlot(A)
    hplotc.SlidingPlot(np.moveaxis(A, -1, 0))
    hplotc.SlidingPlot(np.moveaxis(C, -1, 0) > 0.8*np.max(C))
    t0 = time.time()

    # # # Test to see the actual results... of a training case..
    # # # Yeah it is just bad....
    import model.UNet3D
    import torch
    A = model.UNet3D.Unet3D(num_pool_layers=3)
    dir_state_dict = '/home/bugger/Documents/data/grand_challenge/model_weights/model_weights.pt'
    state_dict = torch.load(dir_state_dict, map_location=torch.device('cpu'))
    A.load_state_dict(state_dict)

    input_file_path = '/home/bugger/Documents/data/grand_challenge/validation/input/127_image.nii.gz'
    target_file_path = '/home/bugger/Documents/data/grand_challenge/validation/target/127_mask.nii.gz'

    input_data = nib.load(input_file_path).get_fdata()
    target_data = nib.load(target_file_path).get_fdata()

    input_data_sel = torch.as_tensor(input_data[125-68:125+68, 250-68:250+68, 165-61:165+61]).float()
    input_data_sel = torch.stack([input_data_sel, input_data_sel, input_data_sel], dim=0)
    with torch.no_grad():
        y_pred = A(input_data_sel[None, ...])

    y_np = y_pred.numpy()[0]
    B = np.exp(y_np) / np.sum(np.exp(y_np), axis=0)
    hplotc.SlidingPlot(np.moveaxis(B, -1, 0))
    C = (B[1] > 0.7).astype(int) + 2 * (B[2] > 0.5).astype(int)


    D = target_data[125 - 22:125 + 22, 250 - 22:250 + 22, 165 - 14:165 + 14]
    hplotc.SlidingPlot(np.moveaxis(C, -1, 0))
    hplotc.SlidingPlot(np.moveaxis(D, -1, 0))

    hplotc.SlidingPlot(np.moveaxis(C, -1, 0))
    hplotc.SlidingPlot(np.moveaxis(target_data[125-68:125+68, 250-68:250+68, 165-61:165+61], -1, 0))
    hplotc.SlidingPlot(np.moveaxis(y_pred[0].numpy(), -1, 0))

    B = np.exp(A) / np.sum(np.exp(A), axis=0)
    hplotc.SlidingPlot(np.moveaxis(B, -1, 0))
    C = (B[1] > 0.99).astype(int) + 2 * (B[2] > 0.5).astype(int)


bone_cold_execution = False
if bone_cold_execution:
    import model.UNet3D
    import os
    import numpy as np
    import torch
    import nibabel as nib

    A = model.UNet3D.Unet3D(in_chans=1, out_chans=1, num_pool_layers=3)
    # dir_state_dict = '/home/bugger/Documents/data/grand_challenge/model_weights/model_weights.pt'
    dir_state_dict = '/data/seb/model_run/unet_validation_20200421_third_iteration/config_00/model_weights.pt'
    device = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(dir_state_dict, map_location=torch.device(device))
    A.load_state_dict(state_dict)

    ddata = '/home/seb/data/grand_challenge/data/final_test/test/input'
    list_files = os.listdir(ddata)
    i_file = os.path.join(ddata, list_files[0])
    print(i_file)
    temp_data = nib.load(i_file).get_fdata()
    img = torch.as_tensor(temp_data[np.newaxis, np.newaxis]).float()
    tens_data = (img - img.min()) / (img.max() - img.min()).float()
    A.to(device)
    tens_data = tens_data.to(device)
    print('Calculating')
    A.eval()
    with torch.no_grad():
        temp_output = A(tens_data[:,:,:,:,0:116])


pad_solution = False
if pad_solution:

    # Load the local data...
    import numpy as np
    import nibabel as nib
    import os
    import helper.plot_class as hplotc
    import matplotlib.pyplot as plt
    import time

    data_dir = '/home/bugger/Documents/data/grand_challenge/prediction/prediction_sigmoid'
    dest_dir = '/home/bugger/Documents/data/grand_challenge/prediction/prediction_sigmoid_padded'
    file_list = [x for x in os.listdir(data_dir) if x.endswith('nii.gz')]
    t0 = time.time()

    for i_file in file_list:
        # i_file = [x for x in file_list if '203' in x][0]
        file = os.path.join(data_dir, i_file)
        dest_path = os.path.join(dest_dir, i_file)
        t1 = time.time()
        print(dest_path, t1 - t0)
        A = nib.load(file).get_fdata()
        A = A.astype(np.float32)
        print(A.shape)
        if A.ndim > 3:
            A = A[0]
        A_shape = A.shape
        B = np.zeros((A_shape[0] + 92, A_shape[1] + 92, A_shape[2] + 94))
        B[:A_shape[0], :A_shape[1], :A_shape[2]] = A
        C = nib.Nifti1Image(B, np.eye(4))
        nib.save(C, dest_path)

    # Inspect the data ovf softmax
    import numpy as np
    import os
    import nibabel as nib

    # dir_data = '/home/bugger/Documents/data/grand_challenge/prediction'
    dir_data = '/home/seb/data/grand_challenge/data/final_test/prediction'
    file_list = [x for x in os.listdir(dir_data) if 'old' not in x]
    i_file = file_list[1]
    file = os.path.join(dir_data, i_file)
    A = nib.load(file).get_fdata()
    hplotc.SlidingPlot(A)
    t0 = time.time()



scale_solution = False
if scale_solution:

    # Load the local data...
    import numpy as np
    import nibabel as nib
    import os
    import helper.plot_class as hplotc
    import matplotlib.pyplot as plt
    import time

    data_dir = '/home/bugger/Documents/data/grand_challenge/prediction/prediction_sigmoid'
    dest_dir = '/home/bugger/Documents/data/grand_challenge/prediction/prediction_sigmoid_scaled_ceiling'
    file_list = [x for x in os.listdir(data_dir) if x.endswith('nii.gz')]
    t0 = time.time()

    for i_file in file_list:
        i_file = [x for x in file_list if '203' in x][0]
        file = os.path.join(data_dir, i_file)
        dest_path = os.path.join(dest_dir, i_file)
        t1 = time.time()
        print(dest_path, t1 - t0)
        A = nib.load(file).get_fdata()
        if A.ndim > 3:
            A = A[0]

        A_shape = A.shape
        A_newshape = (A_shape[0] + 92, A_shape[1] + 92, A_shape[2] + 94)
        from skimage.transform import resize
        bin_1 = A[0, 0, :] != 0.5
        bin_2 = A[:, 0, 0] != 0.5
        bound_1 = np.sum(bin_1)
        bound_2 = np.sum(bin_2)
        A_new = resize(A[:bound_2, :bound_2, :bound_1], A_newshape)
        hplotc.SlidingPlot(A_new)
        C = nib.Nifti1Image(B, np.eye(4))
        nib.save(C, dest_path)

    # Inspect the data ovf softmax
    import numpy as np
    import os
    import nibabel as nib

    # dir_data = '/home/bugger/Documents/data/grand_challenge/prediction'
    dir_data = '/home/seb/data/grand_challenge/data/final_test/prediction'
    file_list = [x for x in os.listdir(dir_data) if 'old' not in x]
    i_file = file_list[1]
    file = os.path.join(dir_data, i_file)
    A = nib.load(file).get_fdata()
    hplotc.SlidingPlot(A)
    t0 = time.time()
