
import h5py
import re
import scipy.io
import helper.array_transf as harray
import numpy as np
import helper.misc as hmisc
import os
import data_prep.registration.RegistrationProcess as RegistrationProcess

"""
Create a protocol that can link these two...
We need the M&M data set
We need Bart's sliced B1 data
"""

# Define the MM database set...
mm_dataset_base = '/media/bugger/MyBook/data/m&m/MnM-2/training'
list_4ch_files = []
for d, _, f in os.walk(mm_dataset_base):
    if len(f):
        print(d)
        filter_f = [os.path.join(d, x) for x in f if 'LA_ED.nii' in x or 'LA_ES.nii' in x]
        list_4ch_files.extend(filter_f)

list_4ch_segm_files = []
for i_file in list_4ch_files:
    dir_name = os.path.dirname(i_file)
    base_name = hmisc.get_base_name(i_file)
    ext = hmisc.get_ext(i_file)
    segm_name = base_name + "_gt" + ext
    list_4ch_segm_files.append(os.path.join(dir_name, segm_name))


# Define the B1 database set
data_type = 'p4ch'
bart_b1_base = f'/media/bugger/MyBook/data/simulated/cardiac/bart/{data_type}'
bart_b1_minus = os.path.join(bart_b1_base, 'b1_minus')
bart_b1_files = os.listdir(bart_b1_minus)
bart_b1_plus = os.path.join(bart_b1_base, 'b1_plus')

sel_b1 = bart_b1_files[0]
bart_b1_minus_file = os.path.join(bart_b1_minus, sel_b1)
bart_b1_plus_file = os.path.join(bart_b1_plus, sel_b1)

A_b1_minus = np.load(bart_b1_plus_file)
A_b1_plus = np.load(bart_b1_plus_file)
import helper.plot_class as hplotc
hplotc.ListPlot([A_b1_plus, A_b1_minus], augm='np.abs', vmin=(0, 0.0000001))

import importlib
# We select one b1m/b1p file and register that to all the selected rho files
regproc_obj = RegistrationProcess.RegistrationProcess(rho_files=list_4ch_files[:4],
                                                      segm_files=list_4ch_segm_files[:4],
                                                      b1m_file=bart_b1_minus_file,
                                                      b1p_file=bart_b1_plus_file,
                                                      dest_path='/home/bugger/Documents',
                                                      data_type=data_type,
                                                      registration_options='rigidaffinebspline',
                                                      n_cores=4)


regproc_obj.get_max_slice(0)
data_container = regproc_obj.run_debug(0, 0)

data_container.keys()
plot_obj = hplotc.ListPlot(list(data_container.values()), augm='np.abs')