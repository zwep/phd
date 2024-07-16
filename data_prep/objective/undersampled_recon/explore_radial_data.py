
"""
Two paths:

undersample cartesian data with spokes.. and recover (This has single coil info)
use radial acq data, undersample it, reconstruct it.. and recover (This has multiple coil info)

This data set is acquired in a gated setting
"""

import os
import numpy as np
import scipy.io
import helper.plot_class as hplotc
import helper.array_transf as harray
import matplotlib.pyplot as plt

ddata = '/media/bugger/MyBook/data/7T_data/cardiac_radial/V9_15934'
list_files = [os.path.join(ddata, x) for x in os.listdir(ddata)]

sel_file = list_files[0]
A = scipy.io.loadmat(sel_file).get('reconstructed_data', None)
if A is None:
    A = scipy.io.loadmat(sel_file).get('kpos_data', None)

    plt.scatter(A[0].ravel()[:420], A[1].ravel()[:420])
else:
    A = np.moveaxis(np.moveaxis(np.squeeze(A), -2, 0), -1, 0)
    hplotc.SlidingPlot(A)
    hplotc.SlidingPlot(harray.transform_kspace_to_image_fftn(A, dim=(-2, -1)))

    # CHeck the cpx data..
    ddata_without_recon = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_03_06/V9_17913/v9_06032021_1206437_7_3_transradialfastV4.cpx'
    import reconstruction.ReadCpx as read_cpx
    cpx_obj = read_cpx.ReadCpx(ddata_without_recon)
    A = cpx_obj.get_cpx_img()
    A.shape
    hplotc.SlidingPlot(A)