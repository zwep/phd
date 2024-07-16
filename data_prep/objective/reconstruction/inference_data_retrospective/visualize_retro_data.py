import numpy as np
from objective_helper.reconstruction import convert_direct2cpx_img, convert_direct2cpx
import helper.plot_class as hplotc
import helper.misc as hmisc
import helper.array_transf as harray
import os


def visualize_single_file(ddir, sel_index=0, ext='h5'):
    file_list = [x for x in os.listdir(ddir) if x.endswith(ext)]
    sel_file = file_list[sel_index]
    file_path = os.path.join(ddir, sel_file)
    temp_array = hmisc.load_array(file_path, data_key='kspace', sel_slice='mid')
    x_array = convert_direct2cpx_img(temp_array)
    kspace_array = convert_direct2cpx(temp_array)
    sort_indices = np.argsort(np.abs(kspace_array).max(axis=(1, 2)))
    # print(x_array.shape)
    plot_obj = hplotc.ListPlot([np.abs(x_array)[sort_indices]], ax_off=True, proper_scaling=True)
    return plot_obj

dretro = '/media/bugger/MyBook/data/7T_data/direct_inference_retro/input'
dtrigger = '/media/bugger/MyBook/data/7T_data/cardiac_radial_us_fs/input'

visualize_single_file(dretro, 1)

visualize_single_file(dtrigger, 1)
