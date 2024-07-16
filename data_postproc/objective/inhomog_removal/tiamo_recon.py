
"""
Here we going to combine two different images to work with virtual coils blabla
https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.22527
"""

import scipy.signal
import numpy as np
import helper.plot_fun as hplotf
import matplotlib.pyplot as plt
import reconstruction.ReadCpx as read_cpx
import torch
import helper.misc as hmisc
import helper.array_transf as harray
import objective.inhomog_removal.executor_inhomog_removal as executor
import pygrappa
import os


def calc_svd(global_array, sel_y, sel_x):
    sel_array = np.take(global_array, sel_x, axis=-1)
    sel_array = np.take(sel_array, sel_y, axis=-1)
    left_x, eig_x, right_x = np.linalg.svd(sel_array, full_matrices=False)
    right_x = right_x.conjugate().T
    return eig_x[0], left_x[:, 0], right_x[:, 0]


"""
Setting directories
"""

main_path = '/media/bugger/MyBook/data/7T_scan/prostate'
measured_path_prostate = os.path.join(main_path, '2020_06_17/ph_10930')
prefix_path_prostate = '/home/bugger/Documents/data/semireal'
model_path_prostate = '/home/bugger/Documents/model_run/inhomog_removal_biasfield/resnet_15_juli'

model_path_2ch = '/home/bugger/Documents/model_run/homog_removal/p2ch_cardiac/gan_ynet'
measured_path_2ch = ''
model_path_4ch = '/home/bugger/Documents/model_run/homog_removal/p4ch_cardiac/gan_ynet'
measured_path_4ch = ''
prefix_path_cardiac = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx'

prediction_mode = 'prostate'

if prediction_mode == 'prostate':
    model_path = model_path_prostate
    prefix_path = prefix_path_prostate
    measured_path = measured_path_prostate
elif prediction_mode == 'p2ch':
    model_path = model_path_2ch
    prefix_path = prefix_path_cardiac
    measured_path = measured_path_2ch
elif prediction_mode == 'p4ch':
    model_path = model_path_4ch
    prefix_path = prefix_path_cardiac
    measured_path = measured_path_4ch
else:
    model_path = ''
    prefix_path = ''

"""
Get the config param
"""
config_param = hmisc.convert_remote2local_dict(model_path, path_prefix=prefix_path)
# Otherwise squeeze will not work properly..
config_param['data']['batch_size'] = 1

"""
Load the model
"""

model_obj = executor.ExecutorInhomogRemovalGAN(config_file=config_param, inference=True, debug=False)
model_obj.load_weights()

"""
Loading files

Creating sel_file_list that we are going to load...
"""

file_list = [os.path.join(measured_path, x) for x in os.listdir(measured_path)]

# Select those images which have a different shim setting..
filter_on = 'cartwfs'
filter_on = 't2w'
sel_file_list = [x for x in file_list if filter_on in x and x.endswith('cpx')]

"""
Load all the selected files - these should all have different shim settings..
"""

loaded_img_files = []
for x in sel_file_list:
    A, A_list = read_cpx.read_cpx_img(x, sel_loc=[0])
    loaded_img_files.append(np.squeeze(A))

n_files = len(loaded_img_files)

# Make sure that we only have 8 coils as input...
sel_coil_img_files = [np.squeeze(x[-8:]) for x in loaded_img_files]

"""
Now try to use GRAPPA on all the possible combinations of loaded files

Conclusion: Using the SoS of hte combination is just as effective..
"""
# Okay now we combine the two kspaces of differnet shim settings
# Now we do the same... but combine the kspace data in half space
plot_intermediate = False
# Number of calibration lines for GRAPPA
n_calib = 4
comb_measured_list = []
for index_i in range(n_files):
    for index_j in range(n_files):
        print(index_i, index_j)
        result_kspace_comb = np.zeros(sel_coil_img_files[index_i].shape)
        n_c, n_y, n_x = result_kspace_comb.shape
        n_mid = int(n_y//2)
        img_index_i = sel_coil_img_files[index_i]
        img_index_j = sel_coil_img_files[index_j]

        # This calculated the kspace combination with GRAPPA
        # But in this setting, there was no difference with the normal approach of combining coils SoS
        # kspace_index_i = harray.transform_image_to_kspace_fftn(img_index_i, dim=(-2, -1))
        # kspace_index_j = harray.transform_image_to_kspace_fftn(img_index_j, dim=(-2, -1))
        # kspace_index_ij = np.concatenate([kspace_index_i, kspace_index_j])
        #
        # Define a calibration kspace measurement
        # calib_index_ij = np.zeros(kspace_index_ij.shape, dtype=np.complex)
        # calib_index_ij[:, n_mid - n_calib: n_mid + n_calib] = kspace_index_ij[:, n_mid - n_calib: n_mid + n_calib]
        # kspace_index_ij[:, ::2, :] = 0
        #
        # combined_kspace = pygrappa.grappa(kspace_index_ij, calib_index_ij, coil_axis=0)
        # combined_kspace_img = harray.transform_kspace_to_image_fftn(combined_kspace, dim=(-2, -1))

        combined_imgspace = np.concatenate([img_index_i, img_index_j], axis=0)

        if plot_intermediate:
            # Dummy variable for now..
            combined_kspace_img = np.zeros((8, 10, 10))
            hplotf.plot_3d_list([combined_kspace_img.sum(axis=0), combined_imgspace.sum(axis=0)], augm='np.abs',
                                subtitle=[['comb kspace '], ['comb img space']])

        # It makes no difference which one we add...
        comb_measured_list.append(combined_imgspace)

"""
Pass all the loaded files through the model
"""

model_output = []
for x_input in sel_coil_img_files:
    x_input = x_input / np.max(np.abs(x_input))
    n_y, n_x = x_input.shape[-2:]
    x_inputed = harray.to_stacked(x_input, cpx_type='cartesian', stack_ax=0)
    x_inputed = x_inputed.T.reshape((n_x, n_y, -1)).T
    A_tensor = torch.as_tensor(x_inputed[np.newaxis]).float()
    with torch.no_grad():
        output = model_obj.generator(A_tensor)

    model_output.append(output[0][0].detach().numpy())


"""
Compare a SoS combination of model output to a normal SoS...
"""

result_model_combination = []
result_sos_combination = []
n_files = len(model_output)
import itertools

for x, y in itertools.product(range(n_files), range(n_files)):
    input_1 = sel_coil_img_files[x]
    input_2 = sel_coil_img_files[y]
    input_comb = np.concatenate([input_1, input_2], axis=0)
    model_output_sos = model_output[x] + model_output[y]
    no_model_output_sos = np.sqrt((np.abs(input_comb) ** 2).sum(axis=0))

    result_model_combination.append(model_output_sos)
    result_sos_combination.append(no_model_output_sos)


hplotf.plot_3d_list(np.stack([x for x in result_model_combination], axis=0)[None], augm='np.abs',
                    subtitle=[list(itertools.product(range(n_files), range(n_files)))],
                    title='Result of model')

hplotf.plot_3d_list(np.stack(result_sos_combination, axis=0)[None], augm='np.abs',
                    subtitle=[list(itertools.product(range(n_files), range(n_files)))],
                    title='Result of SoS')

"""
Process the output, and re-apply to the input..
"""

plot_intermediate = False

sel_image = 3
sel_image = sel_image % len(result_sos_combination)
# Input image
sos_image = result_sos_combination[sel_image]
print(sos_image.min(), sos_image.mean(), sos_image.max(), sos_image.max()/ sos_image.mean())
# Resulting image
result_image = result_model_combination[sel_image]
print(result_image.min(), result_image.mean(), result_image.max(), result_image.max()/result_image.mean())
# Scale the resulting image to the same scale as the measured one
# This makes sure that division/difference will be better...
scaled_result_image = (((result_image - result_image.min()) / (result_image.max() - result_image.min())) * sos_image.max()) + sos_image.min()
print(scaled_result_image.min(), scaled_result_image.mean(), scaled_result_image.max(), scaled_result_image.max()/scaled_result_image.mean())

log_scale_method = False
if log_scale_method:
    diff = np.log(sos_image + 1e-6) - np.log(scaled_result_image + 1e-6)
else:
    diff = sos_image / (scaled_result_image)

if plot_intermediate:
    plt.figure()
    plt.imshow(diff)
    plt.figure()
    plt.hist(diff.ravel(), bins=100)

n_std_meanmax = (diff.max() - diff.mean())/diff.std()
print('Amount of std in max/mean diff ', n_std_meanmax)
n_std_scale = int(n_std_meanmax / 6)
print('Chosen std scale ', n_std_scale)
diff[diff > (diff.mean() + n_std_scale * diff.std())] = 1

if plot_intermediate:
    plt.figure()
    plt.imshow(diff)
    plt.figure()
    plt.hist(diff.ravel(), bins=100)

n_y, n_x = diff.shape
diff_kspace = harray.transform_image_to_kspace_fftn(diff)

# Method 1
kspace_tolerance = False

if kspace_tolerance:
    perc_max = 0.03
    tolerance = np.max(np.abs(diff_kspace)) * perc_max
    tolerance_points = np.abs(diff_kspace) >= tolerance
    plt.imshow(tolerance_points)
    tolerance_mask = harray.convex_hull_image(np.abs(diff_kspace) >= tolerance)
    plt.imshow(tolerance_mask)
    diff_lowfreq = np.abs(harray.transform_kspace_to_image_fftn(tolerance_mask * diff_kspace))
else:
    limit_y = int(0.47 * n_y)
    limit_x = int(0.47 * n_x)
    diff_kspace[:limit_y] = 0
    diff_kspace[-limit_y:] = 0
    diff_kspace[:, :limit_x] = 0
    diff_kspace[:, -limit_x:] = 0
    diff_lowfreq = np.abs(harray.transform_kspace_to_image_fftn(diff_kspace))

hplotf.plot_3d_list([diff, diff_lowfreq, diff - diff_lowfreq])

smoothing_kernel = True

if smoothing_kernel:
    perc_kernel = 0.05
    n_mask = int(perc_kernel * min(diff_lowfreq.shape))
    print('Size of kernel ', n_mask)
    kernel = np.ones((n_mask, n_mask)) / n_mask ** 2
    conv_mode = 'same'
    conv_boundary = 'symm'
    diff_smooth = scipy.signal.convolve2d(diff_lowfreq, kernel, mode=conv_mode, boundary=conv_boundary)
    plt.imshow(diff_smooth)
else:
    diff_smooth = diff_lowfreq


if log_scale_method:
    postproc_image = np.log(sos_image) - diff_smooth
else:
    postproc_image = sos_image / diff_smooth

n_std_meanmax = int((postproc_image.max() - postproc_image.mean())/postproc_image.std())

for n_factor_std in range(n_std_meanmax):
    if log_scale_method:
        postproc_image = np.log(sos_image) - diff_smooth
    else:
        postproc_image = sos_image / diff_smooth

    postproc_image[(postproc_image.mean() + n_factor_std * postproc_image.std()) < postproc_image] = 1
    plot_array_result = np.array([sos_image, result_image, postproc_image])[np.newaxis]
    fig = hplotf.plot_3d_list(plot_array_result, wspace=0, hspace=0,
                              aspect='auto', ax_off=True, subtitle=[['Input', 'Result', 'Post proc']])
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()

