import h5py
import numpy as np
import helper.plot_class as hplotc
import os
import argparse
import helper.misc as hmisc


def file_to_dsp(img):
    sos_img = np.sqrt((np.abs(np.fft.ifftn(img[..., ::2] + 1j * img[..., 1::2], axes=(0, 1))) ** 2).sum(axis=-1))
    return sos_img


def sense_to_dsp(img):
    # sos_img = np.sqrt((np.abs(img[..., 0] + 1j * img[..., 1])).sum(axis=0))
    sos_img = (np.abs(img[..., 0] + 1j * img[..., 1]))[0]
    return sos_img


"""
Baaah
"""

parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, help='Provide the name of the directory that we want to post process')
parser.add_argument('-key', type=str, help='Provide the name of the directory that we want to post process')

p_args = parser.parse_args()
path = p_args.path
key = p_args.key

list_files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('h5')]

img_array = []
for i_file in sorted(list_files):
    if key == 'sense':
        temp_array = hmisc.load_array(i_file, data_key='kspace', sel_slice='mid')
    else:
        temp_array = hmisc.load_array(i_file, data_key=key, sel_slice='mid')
    #
    if key == 'kspace':
        temp_array = file_to_dsp(temp_array)
    if key == 'sense':
        temp_array = sense_to_dsp(temp_array)
        # temp_array = temp_array[..., 0] + 1j * temp_array[..., 1]
    #
    print(temp_array.shape, temp_array.dtype)
    img_array.append(np.abs(temp_array))

plot_obj = hplotc.PlotCollage(img_array, path, n_display=len(img_array), plot_type='array',
                              text_box=False, height_offset=0)
plot_obj.plot_collage()

