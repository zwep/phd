import os
import numpy as np
import helper.plot_class as hplotc
import sys
import helper.misc as hmisc

"""
Something terribly wrong has happened here...

The unfolding files in cartesian_cardiac_cine are wrong
I think the geometry correction, or some rotation has not been done

Therefore the images are warped and dont look good.
Probably I could register them in some way... but I think it is too much work for now.

Note that the short axis images look blurry because of the summing over the z-direction (slices)
"""

ddata_base = '/media/bugger/MyBook/data/7T_data'

# 4ch example
find_file = 'v9_07032021_1247036'
# 2ch example
# find_file = 'v9_02052021_1025205'
# sa example
# find_file = 'ca_28022022_1839558'
# transverse example
# find_file = 'v9_05032022_1252227'
found_file_list = []
for d, _, f in os.walk(ddata_base):
    filter_f = [x for x in f if find_file in x]
    if len(filter_f) > 0:
        for i_file in filter_f:
            ext_file = hmisc.get_ext(i_file)
            if ext_file in ['.png', '.jpg', '.jpeg']:
                continue
            else:
                file_path = os.path.join(d, i_file)
                A = hmisc.load_array(file_path)
                # If the mat-file is returned, which is a dict, then try another key.
                if isinstance(A, dict):
                    A = hmisc.load_array(file_path, data_key='reconstructed_data')
                if ext_file in ['.mat']:
                    A = A.T
                A = np.squeeze(A)
                # To combine the coils...
                if A.ndim == 4:
                    A = np.abs(A).sum(axis=0)
                found_file_list.append((A, d))

array_list, dir_list = zip(*found_file_list)
import importlib
importlib.reload(hplotc)
hplotc.ListPlot([[x[0] for x in array_list]], augm='np.abs', subtitle=[[x for x in dir_list]], hspace=0.4)
hplotc.ListPlot([[x[0] for x in array_list]], augm='np.abs', hspace=0.4)
