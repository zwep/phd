
"""
We now have 2 miages from the clinic... lets check them out
"""

import pydicom
import numpy as np
import os
import helper.plot_class as hplotc

ddata_15 = '/home/bugger/Documents/data/1.5T/prostate_mri_mrl'
ddata_30 = '/home/bugger/Documents/data/3T/prostate_mri_mrl'


def insight_dicom(ddata):
    array_data = []
    counter = 0
    file_list = [x for x in os.listdir(ddata) if x.endswith('dcm')]
    for i_file in file_list:
        counter += 1
        print('\n\nNEW IMAGE ', i_file)
        file_name = os.path.join(ddata, i_file)
        dicom_obj = pydicom.read_file(file_name)
        if hasattr(dicom_obj, 'pixel_array'):
            A = dicom_obj.pixel_array
        else:
            A = np.zeros((100, 100))

        array_data.append(A)

        # Report the imaging parameters
        TR = dicom_obj.get(('0018', '0080'))
        TE = dicom_obj.get(('0018', '0081'))
        slice_thickness = dicom_obj.get(('0018', '0050'))
        pixel_spacing = dicom_obj.get(('0028', '0030'))
        if TR is not None:
            print(dicom_obj[('2001', '100b')].value)
            print('acq matrix', dicom_obj[('0018', '1310')].value)
            print('pixel spacing', pixel_spacing.value)
            print('slice thickness', slice_thickness.value)
            print('rows', dicom_obj[('0028', '0010')].value)
            print('cols', dicom_obj[('0028', '0011')].value)
            print('TE', TE.value)
            print('TR', TR.value)
        else:
            print('We have not found TR')

    return array_data

array_15 = insight_dicom(ddata_15)
hplotc.ListPlot(array_15)
array_30 = insight_dicom(ddata_30)
hplotc.ListPlot(array_30)
# A_15 = pydicom.read_file(file_15).pixel_array
A_30 = pydicom.read_file(file_30).pixel_array
A_30_header = pydicom.read_file(file_30, stop_before_pixels=True)
hplotc.ListPlot([A_30])

