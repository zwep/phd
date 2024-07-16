from objective_configuration.reconstruction import DDATA
import helper.plot_class as hplotc
from objective_helper.reconstruction import convert_direct2cpx
import os
import helper.misc as hmisc
import numpy as np

"""
well.. some examples might contain a lot of noise...

"""

all_array = []
counter = 0
train_path = os.path.join(DDATA, 'mixed', 'train', 'input')
for i_file in os.listdir(train_path):
    if i_file.endswith('h5'):
        base_name = hmisc.get_base_name(i_file)
        file_path = os.path.join(train_path, i_file)
        loaded_array = hmisc.load_array(file_path, data_key='kspace', sel_slice='mid')
        cpx_array = convert_direct2cpx(loaded_array)
        plot_obj = hplotc.ListPlot(np.fft.ifft2(np.fft.ifftshift(cpx_array, axes=(-2, -1))), col_row=(6, 4), cbar=True)
        plot_obj.savefig(base_name)
        hplotc.close_all()
