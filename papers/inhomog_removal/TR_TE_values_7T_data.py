"""
We need some acquisition parameters....

Lets plot them
"""

import numpy as np
import helper.array_transf as harray
import helper.reconstruction as hrecon
import reconstruction.ReadCpx as read_cpx
import os
import re
import pydicom
import itertools
import helper.philips_files as hphilips


def get_minmax_group_key(x, key):
    key_list = [i_group[key] for i_group in x if i_group[key] is not None]
    if len(key_list):
        return min(key_list), max(key_list)
    else:
        return 0, 0

ddata_volunteer = '/media/bugger/MyBook/data/7T_scan/prostate'
ddata_daan = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter'
ddata_bart_7T = '/media/bugger/MyBook/data/multiT_scan/prostaat/7T'
ddata_bart_3T = '/media/bugger/MyBook/data/multiT_scan/prostaat/3T'

# Get parameters of the volunteers
scan_parameters = []
breakers = False
parse_filter_f = False
for d, _, f in os.walk(ddata_volunteer):
    # Check if the directory name contains DICOM and t2w
    if ('t2w' in d.lower()) and d.endswith('DICOM'):
        # Only use images that start with IM, those are the real dicoms
        filter_f = [os.path.join(d, x) for x in f if x.startswith('IM')]
        for i_file in filter_f:
            print("file name", i_file)
            TR, TE = hphilips.get_TR_TE_dicom(i_file)
            slice_thickness, pixel_spacing, FOV = hphilips.get_slice_thickness_pixel_spacing_FOV(i_file)
            temp_dict = {'TR': TR, 'TE': TE, 'slice_thickness': slice_thickness, 'pixel_spacing': pixel_spacing}
            scan_parameters.append(temp_dict)

    # Filter on par/sin files with t2w in their names
    if any(['t2w' in x.lower() for x in f]):
        filter_f = [os.path.join(d, x) for x in f if 't2w' in x and (x.endswith('par') or x.endswith('sin'))]
        for i_file in filter_f:
            i_file_date = re.findall('_([0-9]{8})_', i_file)[0]
            if i_file.endswith('par'):
                print("file name", i_file)
                cpx_obj = read_cpx.ReadCpx(i_file)
                par_file = cpx_obj.get_par_file()
                TR = par_file['Repetition time [msec]']
                TE = par_file.get('echo_time')
                if isinstance(TE, list):
                    TE = TE[0]
                slice_thickness = par_file.get('slice_thickness')
                pixel_spacing = par_file.get('pixel spacing (x,y) (in mm)')
                if isinstance(pixel_spacing, list):
                    pixel_spacing = ' '.join(pixel_spacing)
            elif i_file.endswith('sin'):
                TR = hrecon.get_key_from_sin_file(i_file, 'repetition_times')
                TE = hrecon.get_key_from_sin_file(i_file, 'echo_times')
                slice_thickness = hrecon.get_key_from_sin_file(i_file, 'slice_thickness')
                pixel_spacing = hrecon.get_key_from_sin_file(i_file, 'voxel_sizes')
            else:
                slice_thickness = None
                pixel_spacing = None
                TR = None
                TE = None

            temp_dict = {'TR': TR, 'TE': TE, 'slice_thickness': slice_thickness, 'pixel_spacing': pixel_spacing, 'date':i_file_date}
            scan_parameters.append(temp_dict)

# Filter on if we have a date or not... this way it is easier
scan_parameters_filtered = [x for x in scan_parameters if x.get('date') is not None]
for ii, group in itertools.groupby(scan_parameters_filtered, key=lambda x: x['date']):
    print(ii)
    group = list(group)
    TRmin, TRmax = get_minmax_group_key(group, 'TR')
    print(f'\t TR:            {TRmin}, {TRmax}')
    TEmin, TEmax = get_minmax_group_key(group, 'TE')
    print(f'\t TE:            {TEmin}, {TEmax}')
    pxmin, pxmax = get_minmax_group_key(group, 'pixel_spacing')
    print(f'\t pixel spacing: {pxmin}, {pxmax}')
    stmin, stmax = get_minmax_group_key(group, 'slice_thickness')
    print(f'\t pixel spacing: {stmin}, {stmax}')


# Daan's parameters
TR_list = []
TE_list = []
slice_thickness_list = []
pixel_spacing_list = []
for d, _, f in os.walk(ddata_daan):
    filter_f = [os.path.join(d, x) for x in f if x.endswith('dcm')]
    if filter_f:
        TR, TE = zip(*[hphilips.get_TR_TE_dicom(i_file) for i_file in filter_f])
        st, ps, _ = zip(*[hphilips.get_slice_thickness_pixel_spacing_FOV(i_file) for i_file in filter_f])

        TR_list.append(list(TR))
        TE_list.append(list(TE))
        slice_thickness_list.append(list(st))
        pixel_spacing_list.append(list(ps))

TR_list = list(itertools.chain(*TR_list))
TE_list = list(itertools.chain(*TE_list))
slice_thickness_list = list(itertools.chain(*slice_thickness_list))
pixel_spacing_list = list(itertools.chain(*pixel_spacing_list))
print('Parameters Daan 7T')
print('TR', harray.get_minmeanmediammax([x for x in TR_list if x]))
print('TE', harray.get_minmeanmediammax([x for x in TE_list if x]))
print('ST', harray.get_minmeanmediammax([x for x in slice_thickness_list if x]))
print('PS', harray.get_minmeanmediammax([x for x in pixel_spacing_list if x]))

# Bart's parameters 7T
TR_list = []
TE_list = []
for d, _, f in os.walk(ddata_bart_7T):
    filter_f = [os.path.join(d, x) for x in f if x.endswith('dcm')]
    if filter_f:
        TR, TE = zip(*[hphilips.get_TR_TE_dicom(i_file) for i_file in filter_f])
        TR_list.append(list(TR))
        TE_list.append(list(TE))

TR_list = list(itertools.chain(*TR_list))
TE_list = list(itertools.chain(*TE_list))
print('Parameters Bart 7T')
print('TR', harray.get_minmeanmediammax([x for x in TR_list if x]))
print('TE', harray.get_minmeanmediammax([x for x in TE_list if x]))

# Bart's parameters 3T
TR_list = []
TE_list = []
for d, _, f in os.walk(ddata_bart_3T):
    filter_f = [os.path.join(d, x) for x in f if x.endswith('dcm')]
    if filter_f:
        TR, TE = zip(*[hphilips.get_TR_TE_dicom(i_file) for i_file in filter_f])
        TR_list.append(list(TR))
        TE_list.append(list(TE))

TR_list = list(itertools.chain(*TR_list))
TE_list = list(itertools.chain(*TE_list))
print('Parameters Bart 3T')
print('TR', harray.get_minmeanmediammax([x for x in TR_list if x]))
print('TE', harray.get_minmeanmediammax([x for x in TE_list if x]))
