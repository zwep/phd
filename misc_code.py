import argparse

"""
Testing arparse
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dont-use-feature', default=False, action='store_true', required=False)

p_args = parser.parse_args()

print(p_args)
feature_value = p_args.dont_use_feature
print(f'Status {feature_value}')


# Test XX_ lodaing of dicom
# Busra spectra thing
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os


def get_spec_data(file_path):
    loaded_dicom = pydicom.read_file(file_path)
    byte_values = loaded_dicom.get(('5600', '0020')).value
    # In the data reader by spec2nii they use np.single
    spec_data = np.frombuffer(byte_values, dtype=np.single)
    spec_data_cpx = spec_data[0::2] + 1j * spec_data[1::2]
    return spec_data_cpx


def plot_spec_data(y_cpx, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(np.abs(y_cpx))
    return ax


dd = os.path.expanduser('~/Documents/data/7T/spectra')
file_list = [x for x in os.listdir(dd) if x.startswith('XX')]

fig, ax = plt.subplots(len(file_list))
for ii, i_file in enumerate(file_list):
    file_path = os.path.join(dd, i_file)
    spec_data = get_spec_data(file_path)
    loaded_dicom = pydicom.read_file(file_path)
    n_size = loaded_dicom.SpectroscopyAcquisitionDataColumns
    # In the reader provided by spec2nii they say it should be conjugated...
    spec_data = spec_data.conjugate()
    spec_data = np.fft.ifftshift(spec_data)
    spec_data = np.fft.ifft(spec_data)
    spec_data = np.fft.fftshift(spec_data)
    ax[ii] = plot_spec_data(spec_data[:n_size], ax[ii])
    ax[ii].set_title(i_file)

fig.savefig(os.path.join(dd, 'spectra_plot.png'))

import helper.misc as hmisc
A = hmisc.load_json('/home/bugger/Documents/paper/reconstruction/inference/unet_PRETR_ACQ/metric.json')
B = hmisc.load_json('/home/bugger/Documents/paper/reconstruction/inference/unet_PRETR_ACQ/metric_rebuttal.json')

for i in B['0p']['pretrained']['undersampled']['0x'].keys():
    if 'ssim' in i:
        print(i, np.mean(A['0p']['pretrained']['undersampled']['0x'][i]))