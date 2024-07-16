import helper.misc as hmisc
import os
import numpy as np
import helper.plot_class as hplotc

"""
is it wise to cluster certain images...?
"""

def file_to_dsp(img):
    sos_img = np.sqrt((np.abs(np.fft.ifftn(img[..., ::2] + 1j * img[..., 1::2], axes=(0, 1))) ** 2).sum(axis=-1))
    return sos_img


def sense_to_dsp(img):
    # sos_img = np.sqrt((np.abs(img[..., 0] + 1j * img[..., 1])).sum(axis=0))
    sos_img = (np.abs(img[..., 0] + 1j * img[..., 1]))[0]
    return sos_img


ddata = '/home/bugger/Documents/data/7T/'
sel_file = os.path.join(ddata, os.listdir(ddata)[0])
key = 'kspace'
temp_array = hmisc.load_array(sel_file, data_key=key, sel_slice='mid')
temp_array = file_to_dsp(temp_array)

from skimage.segmentation import slic
segments = slic(temp_array, n_segments=100, compactness=0.05, start_label=1, channel_axis=None)
hplotc.ListPlot([temp_array, segments])