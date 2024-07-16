import pydicom
import os
import helper.plot_class as hplotc
import helper.misc as hmisc
import numpy as np

"""
Important part... create the correct order
"""

from data_prep.dataset.prostate.daan_reesink.order_of_slices import slice_order_dict

# Just create two images... its fine
# Or four....

ddata = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Seb_pred'
ddest = '/home/bugger/Documents/abstract/ISMRM_2022/patient_gif.gif'
biasfield_files = []
uncor_files = []
for d, _, f in os.walk(ddata):
    filter_f = [x for x in f if 'corrected_biasfield_resnet_15_juli' in x]
    filter_f_uncor = [x for x in f if 'uncorrected' in x]
    if len(filter_f):
        temp_file = os.path.join(d, filter_f[0])
        uncor_file = os.path.join(d, filter_f_uncor[0])
        biasfield_files.append(temp_file)
        uncor_files.append(uncor_file)

sel_file = [3]
pydicom_array = [pydicom.read_file(x).pixel_array[:, 160:850] for x in np.array(biasfield_files)[[1, 3]]]
pydicom_array_uncor = [pydicom.read_file(x).pixel_array[:, 160:850] for x in np.array(uncor_files)[[1, 3]]]
# THIs order is for file 7TMRI010
pydicom_array[-1] = pydicom_array[-1][slice_order_dict['7TMRI010']]
pydicom_array = np.concatenate(pydicom_array, axis=-2)
pydicom_array_uncor = np.concatenate(pydicom_array_uncor, axis=-2)
hplotc.SlidingPlot(pydicom_array)
hplotc.SlidingPlot(pydicom_array_uncor)
# Now combine corrected/uncorrected
stacked_array = np.concatenate([pydicom_array_uncor, pydicom_array], axis=-1)
n_slices = pydicom_array.shape[0]
print(pydicom_array.shape[1:])
scale_factor = pydicom_array.shape[-2] / pydicom_array.shape[-1]
nx = 300
ny = int(nx * scale_factor)
hmisc.convert_image_to_gif(stacked_array, output_path=ddest, n_card=n_slices, nx=nx, ny=ny, duration=15/n_slices)


