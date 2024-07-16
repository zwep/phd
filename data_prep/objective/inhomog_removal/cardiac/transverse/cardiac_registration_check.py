

import numpy as np
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import os
import sys
import helper.misc as hmisc

ddata = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/registered/train/target'
dfiles = os.listdir(ddata)

hplotf.close_all()

for ifile in [x for x in dfiles if x.endswith('25.npy')]:
    filename = os.path.join(ddata, ifile)
    A = np.load(filename)
    if A.ndim == 2:
        A = A[np.newaxis]

    hplotf.plot_3d_list(A, augm='np.abs', title=ifile)


ddata = '/home/bugger/Documents/data/semireal/cardiac_simulation_rxtx/registered/train/input'
dfiles = os.listdir(ddata)

for ifile in [x for x in dfiles if '_05_' in x]:
    filename = os.path.join(ddata, ifile)
    A = np.load(filename)
    if A.ndim == 2:
        A = A[np.newaxis]

    hplotf.plot_3d_list(A, augm='np.abs', title=ifile)
