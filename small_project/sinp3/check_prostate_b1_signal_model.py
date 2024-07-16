import scipy.io
import tooling.shimming.b1shimming_single
import tooling.shimming.b1shimming_single as mb1
import pydicom
import helper.array_transf as harray
import reconstruction.ReadCpx as read_cpx
import os
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import numpy as np

"""
We are going to investigate how the B1 Signal model behaves under different circumstances
and compare it with measured data.

Conclusion: shimming did they trick it seems.
"""



"""
Read data and store it somewehere..
"""

input_dir = '/media/bugger/MyBook/data/7T_scan/prostate_mri_mrl/2021_01_06/pr_16289'
input_dir_dicom = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/2021_01_06/pr_16289'
b1_dicom_dir = [os.path.join(input_dir_dicom, x) for x in os.listdir(input_dir_dicom) if 'B1map' in x]
b1_dicom_files = [os.path.join(x, 'DICOM/IM_0002') for x in b1_dicom_dir]

survey_scan_file = os.path.join(input_dir, 'pr_06012021_1639155_7_2_surveyisoV4.cpx')
b1_shim_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if 'b1shim' in x and x.endswith('cpx')]
b1_map_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if 'b1map' in x and x.endswith('cpx')]
t2_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if 't2' in x and x.endswith('cpx')]

# Load survey scan
cpx_obj = read_cpx.ReadCpx(survey_scan_file)
survey_img = cpx_obj.get_cpx_img()
survey_img = np.squeeze(survey_img)

# Load b1 shim series
b1_shim_array = [np.squeeze(read_cpx.ReadCpx(x).get_cpx_img()) for x in b1_shim_files]
# Load b1 map
b1_map_array = [np.squeeze(read_cpx.ReadCpx(x).get_cpx_img()) for x in b1_map_files]
# Load t2w images
t2_array = [np.squeeze(read_cpx.ReadCpx(x).get_cpx_img()) for x in t2_files]
# Load b1 map in Dicom
b1_dicom_obj = [pydicom.read_file(x) for x in b1_dicom_files]
b1_dicom_array = np.array([x.pixel_array for x in b1_dicom_obj])

# Plot T2 array inhomog patterns
hplotc.ListPlot(np.rot90(np.array(t2_array).sum(axis=1), k=1, axes=(-2, -1)), augm='np.abs')

# Scale B1 map to flip angle and test sin()^3 mapping
b1_sel = b1_dicom_array[:, 3]
b1_mask = np.zeros(b1_sel.shape)
n_n, n_y, n_x = b1_sel.shape
y_center, x_center = (n_y // 2, n_x // 2)
delta_x = int(0.05 * n_y)
x_sub = b1_sel[:, y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x]
b1_mask[:, y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x] = 1
x_mean = np.abs(x_sub.mean())
b1_scaled = b1_sel/x_mean
flip_angle = b1_scaled * np.pi/2
# Plot scaled B1 data and sin ^ 3 scaled
hplotf.plot_3d_list([b1_scaled * (b1_mask+1)])
hplotf.plot_3d_list([np.sin(b1_scaled * np.pi/2) ** 3, b1_scaled])


# Apply shimming to the Transmit side of the B1 shim series
b1_transmit = b1_shim_array[0].sum(axis=0)
b1_transmit = np.rot90(b1_transmit, axes=(-2, -1))
hplotc.ListPlot(b1_transmit, augm='np.abs')

b1_mask = np.zeros(b1_transmit.shape[-2:])
n_c, n_y, n_x = b1_transmit.shape
y_center, x_center = (n_y // 2, n_x // 2)
delta_x = int(0.1 * n_y)
b1_mask[y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x] = 1

# hplotf.plot_3d_list([b1_transmit.sum(axis=0).real * (b1_transmit_mask+1)])
shim_obj = tooling.shimming.b1shimming_single.ShimmingProcedure(b1_transmit, b1_mask, relative_phase=True,
                                                                str_objective='signal_se',
                                                                debug=True)
x_opt, final_value = shim_obj.find_optimum()
b1_transmit_shimmed = harray.apply_shim(b1_transmit, cpx_shim=x_opt)
hplotc.ListPlot([b1_transmit_shimmed * (1+b1_mask), b1_transmit.sum(axis=0)], augm='np.abs')

# Scale the shimmed array to 90 degree flip angle
x_sub = b1_transmit_shimmed[y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x]
x_mean = np.abs(x_sub.mean())
# b1_plus_array_scaled = b1_transmit_shimmed/x_mean
b1_plus_array_scaled = b1_transmit.sum(axis=0)/x_mean
hplotc.ListPlot([np.sin(b1_plus_array_scaled * np.pi/10) ** 2], augm='np.abs', vmin=(0, 10))

# Get Sensitivity maps from t2 array
import reconstruction.sensitivity_map as sens_map
kspace_array = harray.transform_image_to_kspace_fftn(t2_array[0], dim=(-2, -1))
espirit_obj = sens_map.EspiritCalib(kspace_array)
b1m_array_espirit = espirit_obj.run()
hplotf.plot_3d_list(b1m_array_espirit.sum(axis=0), augm='np.abs')

# Try to compute the signal based off the SE equation
flip_angle_180 = b1_scaled * np.pi
# Geen idee wat TR/T1 is..
TR = 1
T1 = 1
TE = 1
T2 = 1
E1 = np.exp(-TR/T1)
E11 = np.exp(TE/(2*T1))
E2 = np.exp(TE/T2)
S = b1m_array_espirit.sum(axis=0)
numerator = E2 * np.sin(flip_angle) * (1 - np.cos(flip_angle_180) * E1 - (1- np.cos(flip_angle_180)) * E1 * E11)
denominator = (1 - np.cos(flip_angle) * np.cos(flip_angle_180) * E1)
signal_se = numerator / denominator
hplotc.ListPlot([signal_se, flip_angle])

# Load oriignal flavio data and check signal thing...
# Load numpy file (registered_
target_dir = '/media/bugger/MyBook/data/semireal/prostate_simulation_rxtx/test/target'
temp = os.path.join(target_dir, os.listdir(target_dir)[11])
temp = '/media/bugger/MyBook/data/semireal/prostate_simulation_rxtx/test/target/M03_to_000041_00003_005__32.npy'
b1p_array = np.load(temp)[0]

# Loading a .mat file
b1_dir = "/media/bugger/MyBook/data/simulated/prostate/b1p_b1m_flavio_mat"
b1_file = os.path.join(b1_dir, os.listdir(b1_dir)[1])
b1_dict = scipy.io.loadmat(b1_file)['Model']
b1_plus_array = np.moveaxis(b1_dict['B1plus'][0][0], -1, 0)
# View loaded array
hplotf.plot_3d_list([b1_plus_array, b1_plus_array.sum(axis=0)], augm='np.abs')

# Create an object that has nice methods...
import data_generator.InhomogRemoval as data_gen
dir_data = '/home/bugger/Documents/data/test_clinic_registration/registrated_h5'
data_type = 'test'
data_gen_obj = data_gen.PreComputerShimSettings(ddata=dir_data, dataset_type=data_type,
                                                        objective_shim='signal_se', file_ext='h5',
                                                        shuffle=False)

# Like.... mask creation
center_mask = data_gen_obj.create_random_center_mask(b1_plus_array.shape[-2:])
b1_plus_array = b1_plus_array * np.exp(-1j * np.angle(b1_plus_array[0]))
b1_plus_array = harray.scale_minmax(b1_plus_array, is_complex=True)

hplotc.ListPlot(np.abs(b1_plus_array).sum(axis=0))
import importlib
importlib.reload(mb1)
shimming_obj = mb1.ShimmingProcedure(b1_plus_array, center_mask, opt_method='BFGS',
                                            relative_phase=True,
                                            str_objective='signal_se')
x_opt, final_value = shimming_obj.find_optimum()
print(x_opt)
b1_shimmed = harray.apply_shim(b1_plus_array, cpx_shim=x_opt)
hplotc.ListPlot([b1_shimmed, np.sin(np.pi * np.abs(b1_shimmed)) ** 3], augm='np.abs')
hplotc.ListPlot([b1_shimmed * center_mask, np.sin(np.abs(b1_shimmed * center_mask)) ** 3], augm='np.abs')


np.sum(np.abs(b1_shimmed) * center_mask) / np.sum(center_mask)
flip_angle_map = data_gen_obj.linear_scale_b1(b1_shimmed, mask=center_mask, flip_angle=data_gen_obj.flip_angle)
b1_signal = data_gen_obj.signal_model(flip_angle_map=flip_angle_map)
mean_flip_angle = np.sum(flip_angle_map * center_mask) / np.sum(center_mask)
print("\tMean flip angle in center mask ", mean_flip_angle)
plot_obj = hplotc.ListPlot([np.abs(b1_shimmed) * (1 + center_mask), flip_angle_map, b1_signal], cbar=True)

b1p_array_shimmed = harray.apply_shim(b1p_array, cpx_shim=x_opt)
mean_std = np.abs(b1p_array_shimmed * b1_mask).mean() / np.sum(b1_mask)
#
# b1_shimmed = harray.apply_shim(b1_plus_array, cpx_shim=x_opt)
# flip_angle_map = self.linear_scale_b1(b1_shimmed, mask=center_mask, flip_angle=self.flip_angle)
# b1_signal = self.signal_model(flip_angle_map=flip_angle_map)
#
# final_value_list.append(final_value)
# mean_std_list.append(mean_std)
#
# import matplotlib.pyplot as plt
# plt.plot(final_value_list)
# plt.plot(mean_std_list)
# final_value_list_temp = np.copy(final_value_list)
# mean_std_list_temp = np.copy(mean_std_list)
# plt.plot(final_value_list_temp, 'r')
# plt.plot(mean_std_list_temp, 'r')
#
# print(final_value)
#
# b1p_array_shimmed = harray.apply_shim(b1p_array, cpx_shim=x_opt)
#
# x_sub = b1p_array_shimmed[y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x]
# x_mean = np.abs(x_sub.mean())
# b1p_array_shimmed_scaled = b1p_array_shimmed/x_mean
# zz = np.sin(np.abs(b1p_array_shimmed_scaled * np.pi/2)) ** 3
# hplotf.plot_3d_list([zz])