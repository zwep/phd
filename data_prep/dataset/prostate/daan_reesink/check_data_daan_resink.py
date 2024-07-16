"""
Load/visualize Daan Reesink data

And get the age of the patients
"""


import os
import numpy as np
import helper.plot_class as hplotc
import reconstruction.ReadListData as read_dl
import collections
import pydicom
import os
import helper.philips_files as hphilips

ddata = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/DaanReesink'


age_list = []
for d, _, f in os.walk(ddata):
    filter_f = [x for x in f if x.lower().endswith('t2w')]
    temp_age = []
    if len(filter_f):
        for sel_file in filter_f:
            load_file = os.path.join(d, sel_file)
            pydicom_obj = pydicom.read_file(load_file, stop_before_pixels=True)
            age_value = pydicom_obj.get(('0010', '1010'))
            if age_value is not None:
                temp_age.append(int(age_value.value[:-1]))

    age_list.append(temp_age)

# We've checked mulitple files.. therefore we average it per patient..
age_per_patient = [int(sum(x)/(len(x)+1e-8)) for x in age_list]
age_per_patient = [x for x in age_per_patient if x != 0]
print('Ages', age_per_patient)

TR_TE_list = []
for d, _, f in os.walk(ddata):
    filter_f = [x for x in f if x.lower().endswith('t2w')]
    temp_age = []
    if len(filter_f):
        sel_file = filter_f[0]
        load_file = os.path.join(d, sel_file)
        TR, TE = hphilips.get_TR_TE_dicom(load_file)
        TR_TE_list.append((TR, TE))


print('TR TE values', collections.Counter(TR_TE_list))

# Now check the imaging detail/precision stuf
voxel_sizes = []
for d, _, f in os.walk(ddata):
    filter_f = [x for x in f if x.lower().endswith('t2w')]
    temp_age = []
    if len(filter_f):
        for sel_file in filter_f:
            load_file = os.path.join(d, sel_file)
            slice_thickness, pixel_spacing, FOV = hphilips.get_slice_thickness_pixel_spacing_FOV(load_file)
            if slice_thickness is not None:
                voxel_sizes.append((slice_thickness, pixel_spacing))

print('Voxel size/pixel spacing values', collections.Counter(voxel_sizes))