"""
The Dicoms we had are all single slice, acquiistion etc..

With this script we can sort them, aggregate them into a single .h5 file for later pre/post processing


(0020, 0012) Acquisition Number                  IS: "3"
(0020, 0013) Instance Number                     IS: "18"
(0008, 0030) Study Time                          TM: '100445'
(2001, 107b) [Acquisition Number]                IS: "2"
(0008, 0012) Date                                ...

"""

import itertools
import nrrd
import pydicom
import os
import numpy as np
import h5py


ddir = '/local_scratch/sharreve/mri_data/vanLier_Prostaat_T2'
validation_dir = '/local_scratch/sharreve/mri_data/prostate_h5'
filter_on_field_strength = 'MRL'
shape_list_dict = {}
counter = 0

# Now we are walking over the directory.
# We could also get all the right directories first, and then convert them one by one.
# That change is easily made.
for d, _, f in os.walk(ddir):
    MRI_MRL = os.path.basename(d)
    patient_id = os.path.basename(os.path.dirname(d))
    target_dir = os.path.join(validation_dir, patient_id, MRI_MRL)
    file_list = [x for x in f if x.endswith('.dcm')]
    n_files = len(file_list)
    if MRI_MRL == filter_on_field_strength:
        if n_files > 0:
            counter += 1
            # Create the target directory
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
            print('Start processing: ', MRI_MRL, patient_id)
            # Load the dicom objects WITHOUT pixel array
            dicom_obj_list = [pydicom.read_file(os.path.join(d, x), stop_before_pixels=True) for x in file_list]
            # Get all the dates from the header
            extracted_dates = [x[('0008', '0012')].value for x in dicom_obj_list]
            print('\t\tNumber of files ', len(extracted_dates), '-', '/', n_files)
            print('\t\tNumber of dates ', len(set(extracted_dates)))
            unique_dates = list(set(extracted_dates))
            # Per date... store the proper file order...
            for i_date in unique_dates:
                sel_date_file_list = [file_list[i] for i, x in enumerate(extracted_dates) if x == i_date]
                sel_date_dicom_list = [dicom_obj_list[i] for i, x in enumerate(extracted_dates) if x == i_date]
                print('\t\tDate: ', i_date)
                print('\t\tNumber of files', len(sel_date_file_list))
                # Get all the acquisition numbers and instance numbers from the files in the directory..
                res_acq = [x.get(('0020', '0012'), None) for x in sel_date_dicom_list]
                num_acq = [x.value if x else None for x in res_acq]
                res_inst = [x.get(('0020', '0013'), None) for x in sel_date_dicom_list]
                num_inst = [x.value if x else None for x in res_inst]
                file_list_dict = [{'acq_num': x, 'inst_num': y, 'file_name': k} for k, x, y in zip(sel_date_file_list, num_acq, num_inst)]
                # We need to remove the Nones here
                file_list_dict = [x for x in file_list_dict if x['acq_num'] is not None]
                sorted_file_list_dict = sorted(file_list_dict, key=lambda x: int(x['acq_num']))
                for acq_num, group in itertools.groupby(sorted_file_list_dict, key=lambda x: x['acq_num']):
                    list_group = list(group)
                    print('\t\t Acquisition number', acq_num)
                    print('\t\t Size of instances', len(list_group))
                    # Wow.. this has someone gone well. The x['inst_num'] is not a real string
                    # BUt part from a DICOM object. THerefore sorting on this went OK, eventhough it looked like
                    # a string. Just for the sake of clarity, I make it an int...
                    inst_num_sorted = sorted(list_group, key=lambda x: int(x['inst_num']))
                    temp_dicom_obj = pydicom.read_file(os.path.join(d, inst_num_sorted[0]['file_name']), stop_before_pixels=True)
                    slice_orientation = str(temp_dicom_obj.get(('2001', '100b')).value).lower()
                    sorted_file_list = [x['file_name'] for x in inst_num_sorted]
                    file_name = f'{i_date}_{str(acq_num).zfill(4)}_{slice_orientation}.h5'
                    target_file_path = os.path.join(target_dir, file_name)
                    sorted_array = np.array([pydicom.read_file(os.path.join(d, x)).pixel_array for x in sorted_file_list])
                    hf = h5py.File(target_file_path, 'w')
                    hf.create_dataset('data', data=sorted_array)
                    hf.close()
        else:
            print('For this one we have zero ', d)
    else:
        print(f'Skipping {MRI_MRL} directory')

