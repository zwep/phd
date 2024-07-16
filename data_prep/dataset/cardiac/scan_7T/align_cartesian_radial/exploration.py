import scipy.io
import numpy as np
import helper.plot_class as hplotc
import helper.misc as hmisc
import helper.array_transf as harray
import re
import os

"""
Exploration file on how we are going to align those two bodies...

Steps:

1. Select a subject
2. Select a slice orientation
3. Get both radial and cartesian acquisitions
4. Align somehow

"""


def find_matching_file(ddir, compare_string, re_obj):
    # In a specific directory
    # With a specific string
    # Using a pre-compiled regexp_obj
    file_list_cart = os.listdir(ddir)
    possible_matches = []
    for sel_file_cart in file_list_cart:
        fulldate_cart = re_obj.findall(sel_file_cart)[0].strip("_")
        if fulldate_cart == compare_string:
            temp_file = os.path.join(ddir, sel_file_cart)
            possible_matches.append(temp_file)
    return possible_matches

def select_array(x):
    if x.ndim == 4:
        x = x[0][0]
    elif x.ndim == 3:
        x = x[0]
    return  x

# subdir_string_radial = 'radial_retro_cardiac_cine'
subdir_string_radial = 'radial_trigger_cardiac_cine'
subdir_string_cartesian = 'cartesian_cardiac_cine'

slice_orientation = 'transverse'
slice_orientation_list = ['transverse', '2ch', '4ch', 'sa']
for slice_orientation in slice_orientation_list:
    dir_h5_cart = f'/media/bugger/MyBook/data/7T_data/{subdir_string_cartesian}/h5_data/{slice_orientation}'
    dir_h5_radial = f'/media/bugger/MyBook/data/7T_data/{subdir_string_radial}/h5_data/{slice_orientation}'
    re_date = re.compile('_[0-9]{8}_')
    file_list_cart = os.listdir(dir_h5_cart)
    matched_cart_radial = []
    for sel_file_cart in file_list_cart:
        cart_file_path = os.path.join(dir_h5_cart, sel_file_cart)
        # print("Looking for similar to ", sel_file_cart)
        # Extract the date and name..?
        fulldate_cart = re_date.findall(sel_file_cart)[0].strip("_")
        # print(fulldate_cart)
        matched_files = find_matching_file(dir_h5_radial, fulldate_cart, re_obj=re_date)
        if len(matched_files):
            # print(len(matched_files))
            # radial_basename = os.path.basename(matched_files[0])
            matched_cart_radial.append((os.path.join(dir_h5_cart, sel_file_cart), matched_files[0]))
    print(f'Orientation: {slice_orientation}: {len(matched_cart_radial)}')

    for i_cart, i_radial in matched_cart_radial:
        cart_basename = os.path.basename(i_cart)
        radial_basename = os.path.basename(i_radial)
        cart_array = hmisc.load_array(i_cart)
        radial_array = hmisc.load_array(i_radial)
        cart_array = select_array(cart_array)
        radial_array = select_array(radial_array)
        hplotc.ListPlot([cart_array, radial_array], subtitle=[[cart_basename], [radial_basename]], ax_off=True)
        break
