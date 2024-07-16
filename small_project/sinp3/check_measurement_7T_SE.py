import matplotlib.pyplot as plt
import pydicom
import helper.array_transf as harray
import helper.plot_class as hplotc
import helper.misc as hmisc
import numpy as np
import skimage.transform
import small_project.sinp3.signal_equation as se_signal_eq

ddata_b1 = '/media/bugger/MyBook/data/7T_scan/phantom/2022_02_16/ca_31817/DICOM_21_1_B1map/DICOM/IM_0002'
ddata_tse = '/media/bugger/MyBook/data/7T_scan/phantom/2022_02_16/ca_31817/DICOM_25_1_T2w/DICOM/IM_0002'

# Read TSE file
array_tse = pydicom.read_file(ddata_tse).pixel_array
array_tse = harray.scale_minmax(array_tse)

# Create mask... with tresholding.. wel zo makkelijk
tse_mask_treshold = harray.get_treshold_label_mask(array_tse, treshold_value=0.05 * array_tse.mean(), class_treshold=0.04, debug=True)
# hplotc.ListPlot([array_tse, tse_mask_treshold])


# Read B1 file
dicom_b1 = pydicom.read_file(ddata_b1)
array_b1 = dicom_b1.pixel_array
# Check how the data looks like
# hplotc.ListPlot(array_b1[4:8], cbar=True)
# Select the flip-angle - post-processed data...
# This value, 14.8299, can be found in the dicom. I cant acces the tag though... It is taking too mcuh time
sel_b1 = array_b1[6] / 14.829971313476562
scale_factor = sel_b1.shape[0] / array_tse.shape[0]
tse_mask_treshold_b1_shape = skimage.transform.rescale(tse_mask_treshold, scale=scale_factor, preserve_range=True, anti_aliasing=False)
hplotc.ListPlot([sel_b1 * tse_mask_treshold_b1_shape, tse_mask_treshold_b1_shape])

# Reinact the flip angle map calculation with the two TR maps
# This is the way... but it is annoying on how to get the division right
# TR1 = dicom_b1[('2005', '1030')][0]
# TR2 = dicom_b1[('2005', '1030')][1]
#
# n = TR2 / TR1
# r = array_b1[5] / (array_b1[4]) * tse_mask_treshold_b1_shape
# r_fraction = (r * n - 1) / (n - r) * tse_mask_treshold_b1_shape
# r_fraction = hmisc.correct_inf_nan(r_fraction)
# flip_angle_map = np.arccos(r_fraction)
# hplotc.ListPlot([flip_angle_map])

# Scale the selected b1 to size of the TSE array
scale_factor = array_tse.shape[0] / sel_b1.shape[0]
b1_scaled = skimage.transform.rescale(sel_b1, scale=scale_factor, preserve_range=True, anti_aliasing=False)

min_b1 = np.min(b1_scaled[tse_mask_treshold==1].ravel())
max_b1 = np.max(b1_scaled[tse_mask_treshold==1].ravel())
mean_tse = 2*np.mean(array_tse[tse_mask_treshold==1].ravel())

alpha_range = np.deg2rad(np.arange(min_b1, max_b1, 1) * 0.9)
y_range = mean_tse * np.sin(alpha_range)
y_range_3 = mean_tse * np.sin(alpha_range) ** 3

# Get the signal from the signal equation
TR_se = 10000 * 1e-3
TE_se = 90 * 1e-3  # https://www.nature.com/articles/s41598-019-54880-x
T1_fat = 4000 * 1e-3
T2_fat = 2000 * 1e-3
se_equation = se_signal_eq.get_t2_signal_general(flip_angle=alpha_range, T1=T1_fat, TE=TE_se, TR=TR_se, T2=None, N=17, beta=np.pi)

# Now also display an image generated by the data creation tool...

dir_data = '/home/bugger/Documents/data/test_clinic_registration/registrated_h5'
import data_generator.InhomogRemoval as data_gen
# shim_path = '/media/bugger/MyBook/data/simulated/transmit_flavio'
shim_path = None

generator_options = {"ddata": dir_data, "dataset_type": 'test', "complex_type": 'cartesian',
                     "input_shape": (1, 256, 256),
                     "debug": True,
                     "masked": False,
                     "file_ext": "h5",
                     "lower_prob": 0.0,
                     "relative_phase": True,
                     "objective_shim": 'signal_se',
                     "mask_fraction": 0.05,
                     "flip_angle": np.pi / 2}

gen = data_gen.DataGeneratorInhomogRemovalH5(target_type='biasfield', transform_type='complex', transform_type_target='real', **generator_options)
container = gen.get_data_creation_components(index=0, sel_slice=50)
fa_map_npy = np.rad2deg(container['fa_map'])

mask_npy = container['mask'].numpy()[0]

input_array = np.abs(container['input']).sum(axis=0).numpy()
input_array = harray.scale_minmax(input_array)
input_array_simple = np.abs(container['input_simple']).sum(axis=0).numpy()
input_array_simple = harray.scale_minmax(input_array_simple)

biasfield_array = container['biasfield']
biasfield_array_simple = container['biasfield_simple']

b1p_array = container['b1p_array']
b1p_array_simple = container['b1p_array_simple']
hplotc.ListPlot([input_array, input_array_simple])

# Display both the measurement, as well as the signal equation
fig, ax = plt.subplots()
# ax.scatter((fa_map_npy[mask_npy == 1]).ravel(), (input_array[mask_npy == 1]).ravel(), alpha=0.01, c='r', label='semi realistic 7T')
ax.scatter((fa_map_npy[mask_npy == 1]).ravel(), (biasfield_array_simple[mask_npy == 1]).ravel(), alpha=0.01, c='r', label='semi realistic 7T')
ax.scatter(0.9*(b1_scaled[tse_mask_treshold == 1]).ravel(), (array_tse[tse_mask_treshold == 1]).ravel(), alpha=0.01,
           c='b', label='measurement 7T')
ax.plot(np.arange(min_b1, max_b1, 1) * 0.9, y_range_3, 'g', label='sin ** 3(alpha)')
ax.plot(np.rad2deg(alpha_range), se_equation, 'y', label='Signal equation')
ax.set_xlabel('angle in degree')
ax.set_ylabel('arbitrary intensity')
ax.legend()
ax.set_xlim(0, 180)
ax.set_ylim(0, 1)
