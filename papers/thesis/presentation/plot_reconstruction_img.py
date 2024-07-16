import os

import numpy as np
# import helper.plot_class as hplotc
import helper.misc as hmisc
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
plt.plot([1])


"""

Get image

Undersamle

Plot undersampled kspace
And image space
"""

plt.ion()
DDATA = '/media/bugger/MyBook/data/7T_data/cardiac_radial_us_fs/input/'
for i_file in os.listdir(DDATA):
    file_path = os.path.join(DDATA, i_file)
    if i_file.endswith('4ch.h5'):
        break
        temp_array = hmisc.load_array(file_path, data_key='kspace')
        sos_img = np.sqrt(
            np.abs(np.fft.ifftn(temp_array[0][:, :, ::2] + 1j * temp_array[0][:, :, 1::2], axes=(0, 1))) ** 2).sum(axis=-1)
        plt.figure()
        plt.imshow(sos_img, cmap='gray')
        plt.show()
        plt.figure()
        z = temp_array[0][:, :, ::2] + 1j * temp_array[0][:, :, 1::2]
        plt.imshow(np.abs(z[:, :, 0]), vmin=0, vmax=0.2 * np.max(np.abs(z[:, :, 0])), cmap='grey')


plt.show()