import h5py
import helper.plot_class as hplotc
import numpy as np
import os
import helper.misc as hmisc
import objective.prostate_weighting.recall_prostate_weighting as recall_model

"""
Get the input to the model...
"""


data_type = 'validation'
ddata_base = f'/local_scratch/sharreve/mri_data/prostate_weighting/{data_type}'
ddata_base = f'/home/bugger/Documents/data/3T/prostate/prostate_weighting/{data_type}'
ddata_1p5T = os.path.join(ddata_base, 'input')
ddata_mask = os.path.join(ddata_base, 'mask')
ddata_3T = os.path.join(ddata_base, 'target')
ddata_3T_cor = os.path.join(ddata_base, 'target_corrected')
ddata_3T_cor_n4 = os.path.join(ddata_base, 'target_corrected_N4')


file_list = os.listdir(ddata_3T_cor)
sel_file = file_list[3]
file_name, ext = os.path.splitext(sel_file)
file_1p5T = os.path.join(ddata_1p5T, sel_file)
file_1p5T_mask = os.path.join(ddata_mask, file_name + '_input' + ext)
file_3T = os.path.join(ddata_3T, sel_file)
file_3T_mask = os.path.join(ddata_mask, file_name + '_target' + ext)
file_3T_cor = os.path.join(ddata_3T_cor, sel_file)
file_3T_cor_n4 = os.path.join(ddata_3T_cor_n4, sel_file)

# load 1.5T data
with h5py.File(file_1p5T, 'r') as f:
    array_1p5T = np.array(f['data'])

with h5py.File(file_3T, 'r') as f:
    array_3T = np.array(f['data'])

with h5py.File(file_3T_mask, 'r') as f:
    mask_array_3T = np.array(f['data'])

with h5py.File(file_3T_cor, 'r') as f:
    array_cor = np.array(f['data'])

with h5py.File(file_3T_cor_n4, 'r') as f:
    array_cor_n4 = np.array(f['data'])


hplotc.SlidingPlot(array_1p5T)

hplotc.SlidingPlot(array_3T, title='Original 3T img')
hplotc.SlidingPlot(array_cor, vmin=(0,1), title='Corrected biasfield 3T img')
hplotc.SlidingPlot(array_cor_n4, title='Corrected N4 3T img')

sel_slice = 72
hplotc.ListPlot([[array_3T[sel_slice], array_cor[sel_slice] * mask_array_3T[sel_slice], array_cor_n4[sel_slice] * mask_array_3T[sel_slice]]], subtitle=[['3T img', 'biasfield model', 'N4']],
                vmin=[[(0,np.max(array_3T)), (0, 1), (0, np.max(array_cor_n4))]], cbar=True)

# Select slices of 3T and 1.5T that go together well
sel_slice_3T = 40
sel_slice_1p5T = 47

# load 1.5T data
with h5py.File(file_1p5T, 'r') as f:
    array_1p5T = np.array(f['data'][sel_slice_1p5T])

with h5py.File(file_1p5T_mask, 'r') as f:
    mask_array_1p5T = np.array(f['data'][sel_slice_1p5T])

# load 3T data
with h5py.File(file_3T, 'r') as f:
    array_3T = np.array(f['data'][sel_slice_3T])

# Load 3T corrected
with h5py.File(file_3T_cor, 'r') as f:
    array_3T_cor = np.array(f['data'][sel_slice_3T])

with h5py.File(file_3T_mask, 'r') as f:
    mask_array_3T = np.array(f['data'][sel_slice_3T])


hplotc.ListPlot([array_1p5T, array_3T, array_3T_cor])