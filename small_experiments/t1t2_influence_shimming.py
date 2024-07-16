import numpy as np
import helper.plot_fun as hplotf
import os
import re
import data_generator.InhomogRemoval as data_gen

"""

Small hack-ish experiment to check the influence of shimming on the B1+ data

"""

dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation_t1t2_rxtx'
gen = data_gen.DataGeneratorInhomogRemovalT1T2(ddata=dir_data, complex_type='polar', t1t2='t2',
                                      inhomogeneity=None, debug=True, random_phase=False)

self = gen
index = 0
i_file = self.file_list[index]
i_file = [x for x in self.file_list if '005__36' in x and 'M04' in x][0]

b1_minus_file = os.path.join(self.input_dir, i_file)
b1_plus_file = os.path.join(self.target_dir, i_file)

re_find_ind = re.findall('.*_to_(.*)__([0-9]+)\.npy', i_file)
if re_find_ind:
    temp_file_name, sel_slice = re_find_ind[0]
    patient_id = temp_file_name.split('_')[0]
    if self.debug:
        print('Got patient id ', patient_id)
else:
    patient_id = '-1'
    print('\t\t WARNING: we have not found a file name and slice nr ', re_find_ind)

if self.t1t2_type.lower() == 't1':
    t1_file = [x for x in os.listdir(self.target_dir + '_t1') if patient_id in x][0]
    t1_file = os.path.join(self.target_dir + '_t1', t1_file)
    t1t2_array = np.load(t1_file)
elif self.t1t2_type.lower() == 't2':
    t2_file = [x for x in os.listdir(self.target_dir + '_t2') if patient_id in x][0]
    t2_file = os.path.join(self.target_dir + '_t2', t2_file)
    t1t2_array = np.load(t2_file)
else:
    t1t2_array = []
    print('Derp')

b1_minus_array = np.load(b1_minus_file)
b1_plus_array = np.load(b1_plus_file)

hplotf.plot_3d_list(b1_plus_array, augm='np.abs')

n_chan, n_y, n_x = b1_plus_array.shape
y_center, x_center = (n_y // 2, n_x // 2)
delta_x = int(0.01 * n_y)
b1_plus_sub = np.sum(b1_plus_array, axis=0)[y_center - delta_x:y_center + delta_x,
              x_center - delta_x:x_center + delta_x]
b1_plus_mean = np.abs(b1_plus_sub.mean())

hplotf.plot_3d_list(b1_plus_array.sum(axis=0)[np.newaxis] / b1_plus_mean, augm='np.abs', cbar=True)
shim_tx_amp = [1, 0, 1, 0, 0.5, 1, 1, 1]
shim_tx_degree = [180, 0, -4, -52, 67, 82, -77, -94]
shim_tx_radian = [np.radians(x) for x in shim_tx_degree]

shim_tx = [r * np.exp(1j * phi) for r, phi in zip(shim_tx_amp, shim_tx_radian)]
# Check to see if angles were converted properly
# [np.degrees(np.angle(x)) for x in shim_tx]
b1_plus_array_shimmed = np.einsum("tmn, t -> mn", b1_plus_array, shim_tx)
b1_plus_shimmed_mean = np.abs(
    b1_plus_array_shimmed[y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x].mean())

hplotf.plot_3d_list(b1_plus_array.sum(axis=0)[np.newaxis] / b1_plus_mean, augm='np.abs', cbar=True)
hplotf.plot_3d_list(b1_plus_array_shimmed[np.newaxis] / b1_plus_shimmed_mean, augm='np.abs', cbar=True)

if self.debug:
    print('Subset of b1 plus array, summed')
    print(b1_plus_sub)
    print('Mean of subset ')
    print(b1_plus_mean)

# Taking the absolute values to make sure that values are between 0..1
# B1 plus interference by complex sum. Then using abs value to scale
b1_plus_abs_scaled = np.abs(np.sin(np.abs(b1_plus_array.sum(axis=0)) / b1_plus_mean * np.pi / 2) ** 2)
b1_plus_shimmed_abs_scaled = np.abs(np.sin(np.abs(b1_plus_array_shimmed) / b1_plus_shimmed_mean * np.pi / 2) ** 2)

# Shows difference between shimmed scaled and non shimmed scaled b1+ distributions
hplotf.plot_3d_list([b1_plus_abs_scaled, b1_plus_shimmed_abs_scaled])

b1_plus_scaled = b1_plus_abs_scaled * np.exp(1j * np.angle(b1_plus_array.sum(axis=0)))
b1_plus_shimmed_abs_scaled = b1_plus_shimmed_abs_scaled * np.exp(1j * np.angle(b1_plus_array.sum(axis=0)))

input_array = t1t2_array * b1_minus_array * b1_plus_scaled
input_array_shimmed = t1t2_array * b1_minus_array * b1_plus_shimmed_abs_scaled

# Shows difference between shimmed scaled and non shimmed scaled b1+ distributions
hplotf.plot_3d_list(input_array.sum(axis=0, keepdims=True), augm='np.abs', title='Not shimmed')
hplotf.plot_3d_list(input_array_shimmed.sum(axis=0, keepdims=True), augm='np.abs', title='Shimmed')
