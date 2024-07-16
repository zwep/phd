import helper.plot_class as hplotc
import numpy as np
import os
import sys
import helper.misc as hmisc
import helper.array_transf as harray
from objective_configuration.reconstruction import DDATA

"""
Visualize the input folders
"""
# If it is none.. or empty...
if not DDATA:
    DDATA = '/media/bugger/MyBook/data/7T_data/cardiac_radial_us_fs/target'

for d, _, f in os.walk(DDATA):
    filter_f = [x for x in f if x.endswith('h5')]
    ddest = os.path.join(os.path.dirname(d), 'PNG')
    if len(filter_f):
        if not os.path.isdir(ddest):
            os.makedirs(ddest)
        print('========================= ')
        print(f'       {d}')
        print('========================= ')
        loaded_array = []
        for sel_file in filter_f:
            sel_path = os.path.join(d, sel_file)
            temp_array = hmisc.load_array(sel_path, data_key='kspace')
            sos_img = np.sqrt(np.abs(np.fft.ifftn(temp_array[0][:, :, ::2] + 1j * temp_array[0][:, :, 1::2], axes=(0,1))) ** 2).sum(axis=-1)
            loaded_array.append(sos_img)
        plot_obj = hplotc.PlotCollage(loaded_array, ddest=ddest, plot_type='array', n_display=len(loaded_array) // 3)
        plot_obj.plot_collage()
