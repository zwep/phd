
import helper.plot_class as hplotc
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import reconstruction.ReadCpx as read_cpx
import reconstruction.SenseUnfold as sense_unfold
import helper.array_transf as harray
import os
import re
import data_prep.unfolding_data.ProcessVnumber as proc_vnumber
import pydicom
import nrrd
import scipy.ndimage

"""
Test to see if we can create a reliable mask using the refscan itself...
"""


def get_v_number(path):
    scan_files = {}
    for d, sd, f in os.walk(path):
        regex_vnumber = re.findall('V9_[0-9]*', d)
        if regex_vnumber:
            v_number = regex_vnumber[0]
            scan_files.setdefault(v_number, [])

    return scan_files


scan_dir = '/media/bugger/MyBook/data/7T_scan/cardiac'
target_dir = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac'

vnumber_dict = get_v_number(scan_dir)

# This is how we get all them v-numbers...
unique_v_numbers = list(sorted(vnumber_dict.keys()))[13:]
v_number_counter = 0

v_number_counter += 1
v_number = unique_v_numbers[v_number_counter]
print('Processing v number ', v_number)
proc_obj = proc_vnumber.ProcessVnumber(v_number, scan_dir=scan_dir, target_dir=target_dir, debug=False,
                                       status=True, save_format='jpeg')
# Load the ref file str
n_ref_files = len(proc_obj.ref_file_str)

ref_file_counter = 1
index, ref_file = proc_obj.ref_file_str[ref_file_counter]
ref_file_no_ext = os.path.splitext(ref_file)[0]
cpx_obj_ref = read_cpx.ReadCpx(ref_file)
res = cpx_obj_ref.get_cpx_img()
object_to_mask = np.squeeze(np.abs(res).sum(axis=0)[0])
# object_to_mask = np.moveaxis(object_to_mask, 2, 0)

a, b = np.histogram(object_to_mask.ravel(), bins=100)
temp_tresh_value = b[2]
full_mask = np.array([harray.get_treshold_label_mask(x, class_treshold=0.02, treshold_value=temp_tresh_value) for x in object_to_mask])
hplotc.SlidingPlot(full_mask)

nrrd.writer.write('/home/bugger/Documents/data/itk_masking/full_refscan.nrrd', object_to_mask.astype(np.float32))
nrrd.writer.write('/home/bugger/Documents/data/itk_masking/full_mask_refscan.nrrd', full_mask.astype(np.uint16))
# Now move to ITK SNAP...

# Now load it again...
full_mask, _ = nrrd.read('/home/bugger/Documents/data/itk_masking/full_mask_refscan.nrrd')

# Per slice... because 3D holes are less common.
# Chose this orientation because that is IMO the most important one.
temp = np.array([scipy.ndimage.binary_fill_holes(x) for x in np.moveaxis(full_mask, 2, 0)])
full_mask = np.moveaxis(temp, 0, 2)
# Inspect the mask...
hplotc.SlidingPlot(np.moveaxis(full_mask, 2, 0))

# Store the newly created mask.....
np.save(ref_file_no_ext, full_mask)