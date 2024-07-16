"""
Unfolding went wrong...

I think I know what the error is..
"""


import helper.plot_class as hplotc
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import scipy.ndimage
import time
import importlib
import os
import warnings
import pandas as pd
import numpy as np
import reconstruction.ReadCpx as read_cpx
import reconstruction.SenseUnfold as sense_unfold
import helper.array_transf as harray
from pandas_ods_reader import read_ods
import re
import data_prep.unfolding_data.ProcessVnumber as proc_vnumber

scan_dir = '/media/bugger/MyBook/data/7T_scan/cardiac'
target_dir = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac'

v_number = 'V9_13975'
proc_obj = proc_vnumber.ProcessVnumber(v_number, scan_dir=scan_dir,
                                       target_dir=target_dir, debug=True,
                                       status=True, save_format='npy')

res = proc_obj.run_single_file(sel_file_name='p2ch')
hplotc.SlidingPlot(res)
hplotc.ListPlot(res, augm='np.abs')