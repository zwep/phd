
"""
Show some..
"""

import os
import numpy as np
import helper.plot_class as hplotc

# Get some input data....

main_data_path = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/'
measured_path = os.path.join(main_data_path, 't2w')
sel_index = 0
sel_file = os.listdir(measured_path)[sel_index]
dir_file = os.path.join(measured_path, sel_file)
A = np.load(dir_file)
hplotc.ListPlot(A, augm='np.abs', vmin=(500, 4000), ax_off=True)
hplotc.ListPlot(np.abs(A).sum(axis=0), augm='np.abs', vmin=(500, 4000), ax_off=True)


"""
Show more prostate_mri_mrl images
"""
ddata = '/media/bugger/MyBook/data/prostatemriimagedatabase'
nrrdfiles = [os.path.join(ddata, x) for x in os.listdir(ddata) if x.endswith('nrrd')]
import nrrd
res = []
for x in nrrdfiles:
    A, _ = nrrd.read(x)
    _, _, n_slice = A.shape
    res.append(np.rot90(A[:, :, int(n_slice/1.2)], k=3))

hplotc.ListPlot(np.array(res)[None], ax_off=True)
