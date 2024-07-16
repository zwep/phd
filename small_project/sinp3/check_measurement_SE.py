import os
import matplotlib.pyplot as plt
import helper.array_transf as harray
import numpy as np
import helper.plot_class as hplotc
import pydicom

ddata = '/media/bugger/B7DF-4571/1p5T/DICOM_SE'
if ddata.endswith('SE'):
    TURBO = ''
else:
    TURBO = 'Turbo'

filter_file_list = sorted([os.path.join(ddata, x) for x in os.listdir(ddata) if x.startswith('IM')])

all_arrays = []
for i_file in filter_file_list:
    temp_obj = pydicom.read_file(i_file)
    pulse_seq = str(temp_obj.get(('0018', '9005')).value)
    flip_angle = int(temp_obj.get(('2001', '1023')).value)
    print(f'Filename: {i_file} --- Sequence type {pulse_seq}')
    temp_array = np.array(temp_obj.pixel_array)
    all_arrays.append((temp_array, flip_angle))

all_arrays = sorted(all_arrays, key=lambda tup: int(tup[1]))
sorted_pixel_array, sorted_flip_angles = zip(*all_arrays)
sorted_pixel_mask = np.array([harray.get_treshold_label_mask(x) for x in sorted_pixel_array])
masked_arrays = sorted_pixel_array * sorted_pixel_mask
hplotc.SlidingPlot(masked_arrays)

mean_signal = np.mean(np.abs(masked_arrays), axis=(-2,-1))
plt.figure()
plt.title(f'{TURBO} Spin echo sequence')
plt.plot(sorted_flip_angles, mean_signal, label='average measured signal')
plt.plot(sorted_flip_angles, np.max(mean_signal) * np.sin(np.deg2rad(sorted_flip_angles)), 'k', label='sin(alpha)')
plt.plot(sorted_flip_angles, np.max(mean_signal) * np.sin(np.deg2rad(sorted_flip_angles)) ** 3, 'r', label='sin(alpha) ** 3')
plt.legend()
