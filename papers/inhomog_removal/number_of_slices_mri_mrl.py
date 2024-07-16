"""
In my paper I say that I have a certain slice range...

I know that this one was based on a previous link between MRI and MRL I guess

Lets re-calculate it based on the registrated data that I have, because I dont realy need t

"""

import re
import os
import h5py
import numpy as np
import pydicom
import collections
import helper.philips_files as hphilips

ddata_vanlier = '/local_scratch/sharreve/mri_data/vanLier_Prostaat_T2'
ddata_h5 = '/local_scratch/sharreve/mri_data/prostate_h5'
re_mri_extracter = re.compile(".*to_([0-9]*)_MR.*")

mri_size_dict = {}
file_name_mapping = {}
for data_type in ['train', 'test', 'validation']:
    print(data_type)
    # Use mask, similar dimension as target_clean, but faster to load
    ddata = f'/local_scratch/sharreve/mri_data/registrated_h5/{data_type}/mask'
    slice_list = []
    file_list = sorted(os.listdir(ddata))
    for i_file in file_list:
        file_name_mapping.setdefault(i_file, [])
        re_mri_mrl_name = re.findall(".*_to_([0-9]*)_MR_(.*).h5", i_file)
        # Get the h5 name..
        if len(re_mri_mrl_name) == 1:
            MR_id, MR_filename = re_mri_mrl_name[0]
            filename_h5 = os.path.join(ddata_h5, MR_id + "_MR", 'MRL', MR_filename + ".h5")
            file_name_mapping[i_file].append([filename_h5])
            with h5py.File(filename_h5, 'r') as f:
                n_slices_h5 = f['data'].shape[0]
            print(f'N slices h5 {n_slices_h5}')
        else:
            print('Uh oh... ', i_file)
        # Get the matching dicom files
        if len(re_mri_mrl_name) == 1:
            MR_id, MR_filename = re_mri_mrl_name[0]
            extracted_date, extracted_acq_num, extracted_slice_orientation = MR_filename.split("_")
            file_path_dcm = os.path.join(ddata_vanlier, MR_id + "_MR", 'MRL')
            file_list_dcm = [x for x in os.listdir(file_path_dcm) if x.endswith('dcm')]
            matching_dcm_files = []
            for i_dcm_file in file_list_dcm:
                dcm_path = os.path.join(file_path_dcm, i_dcm_file)
                pydicom_obj = pydicom.read_file(dcm_path, stop_before_pixels=True)
                acq_nr = str(pydicom_obj.get(('0020', '0012')).value).zfill(4)
                slice_orientation = str(pydicom_obj.get(('2001', '100b')).value).lower()
                date = pydicom_obj.get(('0008', '0012')).value
                if (acq_nr == extracted_acq_num) and (extracted_date == date) and (extracted_slice_orientation == slice_orientation):
                    matching_dcm_files.append(i_dcm_file)
            print('N slices nr of dicom files ', len(matching_dcm_files))
            file_name_mapping[i_file].append([matching_dcm_files])
            # Now get content from the dicom slices...
            header_dict = {'FOV': [], 'n_slice': [], 'pixel_spacing': [], 'slice_thickness': [],
                           'instance_nr': []}
            for x in matching_dcm_files[0:1]:
                dicom_obj = pydicom.read_file(os.path.join(file_path_dcm, x), stop_before_pixels=True)
                res_inst = dicom_obj.get(('0020', '0013')).value
                n_slice = dicom_obj.get(('2001', '1018')).value
                pixel_spacing = dicom_obj.get(('0028', '0030')).value
                slice_thickness = dicom_obj.get(('0018', '0050')).value
                FOV = None
                if len(pixel_spacing) == 2:
                    nrows = dicom_obj.get(('0028', '0010')).value
                    ncols = dicom_obj.get(('0028', '0011')).value
                    FOV = (pixel_spacing[0] * nrows, pixel_spacing[1] * ncols, slice_thickness * n_slice)
                header_dict['FOV'].append(FOV)
                header_dict['n_slice'].append(n_slice)
                header_dict['pixel_spacing'].append(tuple(pixel_spacing))
                header_dict['slice_thickness'].append(slice_thickness)
                # Eigenlijk staan er dus allemaal dubbele in... vet kut...
                # header_dict['instance_nr'].append(int(res_inst))
            for key, item in header_dict.items():
                header_dict[key] = list(set(item))
            # Store the collected information in the mri_size_dict....
            mri_size_dict.setdefault(MR_id, [])
            mri_size_dict[MR_id].append(header_dict)
        else:
            print('Uh oh... ', i_file)


for mri_id in sorted(mri_size_dict.keys(), key=lambda x: int(x)):
    print(mri_id)
    print("\tFOV", min([i_item['FOV'] for i_item in mri_size_dict[mri_id]]))
    print("\tFOV", max([i_item['FOV'] for i_item in mri_size_dict[mri_id]]))
