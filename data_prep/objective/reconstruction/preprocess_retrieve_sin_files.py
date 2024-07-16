import numpy as np
import shutil
import helper.misc as hmisc
import os

"""
We might get away with radial sampling..
We now need to find each .sin file for each fully sampled radial file
That allows us to re-create the spokes that were used

We will use those spokes to project the data onto
Then expand along one axis per spoke
So that a multiplicative mask can take care of the spoke selection
"""

from objective_configuration.reconstruction import ANATOMY_LIST
dbase = '/media/bugger/MyBook/data/7T_data/'
dscan = '/media/bugger/MyBook/data/7T_scan/cardiac'
dsub = 'radial_dataset_'
ddest = '/media/bugger/MyBook/data/7T_data/sin_files_radial'


for i_anatomy in ANATOMY_LIST:
    sub_dir = os.path.join(dbase, dsub + i_anatomy)
    for d, _, f in os.walk(sub_dir):
        filter_f = [x for x in f if x.endswith('.npy')]
        if len(filter_f):
            for i_file in filter_f:
                file_date_time = i_file[:19]
                res = hmisc.find_file_in_dir(file_date_time, dir_name=dscan, ext='sin')
                if res is None:
                    print('Not found ', i_file)
                else:
                    res_base = hmisc.get_base_name(res)
                    res_ext = hmisc.get_ext(res)
                    dtarget = os.path.join(ddest, res_base + res_ext)
                    shutil.copy(res, dtarget)


"""
Check what the content is...
--> Only a small number of files have the same number of spokes
--> We need to adapt to this then
"""

import helper.reconstruction as hrecon
spoke_list = []
for i_file in os.listdir(ddest):
    i_path = os.path.join(ddest, i_file)
    traj = hrecon.get_trajectory_sin_file(i_path)
    n_spokes, n_points, ndim = traj.shape
    spoke_list.append(n_spokes)

import collections
collections.Counter(spoke_list).most_common()