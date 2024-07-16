import numpy as np
import helper.plot_class as hplotc
import os
import helper.misc as hmisc
import helper.array_transf as harray

from objective_configuration.thesis import DPLOT

dd = os.path.join(DPLOT, 'Comparisson Field strength/Prostate')
os.makedirs(dd, exist_ok=True)

dclinic = '/local_scratch/sharreve/mri_data/prostate_h5/'
ddaan = '/local_scratch/sharreve/mri_data/daan_reesink/image'
ddaan_mask = '/local_scratch/sharreve/mri_data/daan_reesink/mask'

patient_list = os.listdir(dclinic)
plot_array = []
mask_array = []
sel_patient = patient_list[0]

patient_dir = os.path.join(dclinic, sel_patient)
patient_dir_1p5T = os.path.join(patient_dir, 'MRL')
patient_dir_3T = os.path.join(patient_dir, 'MRI')

patient_file_1p5T = [os.path.join(patient_dir_1p5T, x) for x in os.listdir(patient_dir_1p5T) if 'trans' in x][0]
patient_file_3T = [os.path.join(patient_dir_3T, x) for x in os.listdir(patient_dir_3T) if 'trans' in x][0]

array_1p5T = hmisc.load_array(patient_file_1p5T)
array_3T = hmisc.load_array(patient_file_3T)
n_slice = array_1p5T.shape[0]

plot_array.append(array_1p5T[n_slice // 2][None])
mask_array.append(harray.get_treshold_label_mask(array_1p5T[n_slice // 2]) == 0)
plot_array.append(array_3T[n_slice // 2][None])
mask_array.append(harray.get_treshold_label_mask(array_3T[n_slice // 2]) == 0)

file_list_daan = os.listdir(ddaan)
sel_file_daan = os.path.join(ddaan, file_list_daan[4])
sel_file_daan_mask = os.path.join(ddaan_mask, file_list_daan[4])
array_7T = hmisc.load_array(sel_file_daan)
n_slice = array_7T.shape[0]
array_7T_mask = hmisc.load_array(sel_file_daan_mask)
plot_array.append(array_7T[n_slice//2][None])
mask_array.append(array_7T_mask[n_slice//2] == 0)

plot_array_masked = [np.ma.masked_array(x, mask=xmask) for x, xmask in zip(plot_array, mask_array)]

plot_obj = hplotc.ListPlot(plot_array_masked, proper_scaling=True, col_row=(3, 1), ax_off=True, wspace=0)
plot_obj.savefig(os.path.join(dd, 'test'), home=False)
