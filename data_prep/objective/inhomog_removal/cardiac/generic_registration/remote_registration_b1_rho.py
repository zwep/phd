
import h5py
import re
import scipy.io
import helper.array_transf as harray
import numpy as np
import helper.misc as hmisc
import os
import data_prep.registration.RegistrationProcess as RegistrationProcess
import skimage.transform as sktransf

"""
Create a protocol that can link these two...
We need the M&M data set
We need Bart's sliced B1 data
"""

dest_dir = '/data/seb/semireal/cardiac_simu_segm_4ch_h5'

hmisc.create_datagen_dir(dest_dir, type_list=('test', 'train', 'validation'),
                         data_list=('input', 'target', 'mask', 'target_clean', 'target_segm'))

# Define the B1 database set
cardiac_type = '4ch'
bart_b1_base = f'/data/seb/simulation/cardiac/b1/{cardiac_type}'
bart_b1_minus = os.path.join(bart_b1_base, 'b1_minus')
bart_b1_plus = os.path.join(bart_b1_base, 'b1_plus')
bart_b1_files = sorted(os.listdir(bart_b1_minus))

b1_dict_split = hmisc.create_train_test_val_files(bart_b1_files, train_perc=0.8, validation_perc=0.1, test_perc=0.1)

# Define the MM database set...
for data_type in ['train', 'test', 'validation']:
    mm_dataset_input = f'/data/seb/mm_segmentation/{data_type}/input'
    mm_dataset_target = f'/data/seb/mm_segmentation/{data_type}/target'
    list_4ch_files = [os.path.join(mm_dataset_input, x) for x in os.listdir(mm_dataset_input) if 'LA_ED.nii' in x or 'LA_ES.nii' in x]
    list_4ch_segm_files = []
    for i_file in list_4ch_files:
        base_name = hmisc.get_base_name(i_file)
        ext = hmisc.get_ext(i_file)
        segm_name = base_name + "_gt" + ext
        list_4ch_segm_files.append(os.path.join(mm_dataset_target, segm_name))


    sel_b1_files = b1_dict_split[data_type]

    # Should be a for loop over Bart's data
    for sel_b1 in sel_b1_files:
        bart_b1_minus_file = os.path.join(bart_b1_minus, sel_b1)
        bart_b1_plus_file = os.path.join(bart_b1_plus, sel_b1)

        n_items = None
        # We select one b1m/b1p file and register that to all the selected rho files
        regproc_obj = RegistrationProcess.RegistrationProcess(rho_files=list_4ch_files[:n_items],
                                                              segm_files=list_4ch_segm_files[:n_items],
                                                              b1m_file=bart_b1_minus_file,
                                                              b1p_file=bart_b1_plus_file,
                                                              dest_path=dest_dir,
                                                              data_type=data_type,
                                                              registration_options='rigidaffinebspline',
                                                              n_cores=16)

        regproc_obj.run()
