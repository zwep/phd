import os
import numbers
import pydicom
import re
import helper.philips_files as hphilips
import numpy as np

"""
The difficulty here is that

we have selected some data that we want to inspect the TR and TE of

however we have converted this data to the h5 format and.... therefore TR and TE values are being forgotten

We need to traceback traceback the original file and get the TR and TE value back

This will take some time, we need the date to gte
"""

base_dir = '/local_scratch/sharreve/mri_data/registrated_h5'
orig_dir = '/local_scratch/sharreve/mri_data/vanLier_Prostaat_T2'

TR_TE_value_dict = {}
slice_thickness_pixel_space_FOV_dict = {}
for data_type in ['train', 'test', 'validation']:
    # Loop over train/test/validation
    input_dir = os.path.join(base_dir, data_type, 'input')
    registrated_files = os.listdir(input_dir)
    # Get the part of the file name that contains the patient ID and the date
    patient_id_date = [re.findall("to_(.*).h5", x)[0] for x in registrated_files]
    unique_patient_id_date = set(patient_id_date)
    unique_patient_id_date = [x.split("_MR_") for x in unique_patient_id_date]
    for patient_id, patient_date in unique_patient_id_date:
        patient_date, _, _ = patient_date.split("_")
        print("\n\nProcessing patient ID", patient_id)
        print("Processing patient date", patient_date)
        # Get all file names
        # Filter based on last part
        # Select on unique
        # Loop over these unique files
        # Get patient ID
        # TODO we can change this to MRI and get the other files..?
        # patient_dir = os.path.join(orig_dir, patient_id + "_MR", 'MRL')
        patient_dir = os.path.join(orig_dir, patient_id + "_MR", 'MRI')
        # Get all patient file names
        patient_files = os.listdir(patient_dir)
        patient_files = [x for x in patient_files if x.endswith('dcm')]
        # Now... we need to get the date from each file... and check if it aligns with the date in the selected file..
        # This snippet below gets the header and extracts the date...
        filter_dicom_header = []
        selected_patient_file = None
        # Misschien een while gebruiken...
        print(len(patient_files))
        for i_patient_file in patient_files:
            dicom_obj = pydicom.read_file(os.path.join(patient_dir, i_patient_file), stop_before_pixels=True)
            # Get all the dates from the header
            extracted_date = dicom_obj[('0008', '0012')].value
            if extracted_date == patient_date:
                print("\t Found date ", extracted_date)
                selected_patient_file = i_patient_file
                break
        # Zoiets als dit....
        if selected_patient_file is None:
            print("No appropriate file found. Are we doing MRI or MRL stuff?")
            selected_patient_file = patient_files[0]
        # I know.. the line above IS cheating
        patient_file_path = os.path.join(patient_dir, selected_patient_file)
        TR_TE_value = hphilips.get_TR_TE_dicom(patient_file_path)
        st_pixel_FOV_value = hphilips.get_size_dict(patient_file_path)
        print("\t Found TR/TE ", TR_TE_value)
        print("\t Found Slice Thickness/Pixel spacing ", st_pixel_FOV_value)
        # Ongeveer dit? Misschie nog avg?
        TR_TE_value_dict[patient_id] = TR_TE_value
        slice_thickness_pixel_space_FOV_dict[patient_id] = st_pixel_FOV_value


TR_list, TE_list = zip(*TR_TE_value_dict.values())
TR_list = [x for x in TR_list if isinstance(x, numbers.Number)]
TE_list = [x for x in TE_list if isinstance(x, numbers.Number)]

import helper.misc as hmisc
hmisc.listdict2dictlist(slice_thickness_pixel_space_FOV_dict)
prep_dict = [list(x.values()) for x in slice_thickness_pixel_space_FOV_dict.values()]
sl_list, pixel_space_list, FOV_list, nslices_list = zip(*prep_dict)


print('Min TR ', min(TR_list))
print('Average TR ', sum(TR_list) / len(TR_list))
print('Max TR ', max(TR_list))
print("")
print('Min TE ', min(TE_list))
print('Average TE ', sum(TE_list) / len(TE_list))
print('Max TE ', max(TE_list))

print('Min slice thickness ', min(sl_list))
print('Average slice thickness ', sum(sl_list) / len(sl_list))
print('Max slice thickness ', max(sl_list))
print("")

pixel_space_list = np.array(pixel_space_list)
print('Min pixel spacing ', np.min(pixel_space_list, axis=0))
print('Average pixel spacing ', np.sum(pixel_space_list, axis=0) / len(pixel_space_list))
print('Max pixel spacing ', np.max(pixel_space_list, axis=0))

FOV_array = np.array(FOV_list)
print('Min pixel spacing ', np.min(FOV_array, axis=0))
print('Average pixel spacing ', np.sum(FOV_array, axis=0) / len(FOV_array))
print('Max pixel spacing ', np.max(FOV_array, axis=0))
