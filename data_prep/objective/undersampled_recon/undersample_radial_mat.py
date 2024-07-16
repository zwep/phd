
"""
We have scanned some patients...

Loaded data wtih ReconFrame.. now we are going to try to undersample this stuff and see how good our
predictions are compared to the ones posed by Reconframe

"""

import matplotlib.pyplot as plt
from pynufft import NUFFT_cpu
import scipy.io
import helper.plot_fun as hplotf

data_path = '/media/bugger/UBUNTU 20_0/data/v9_24012021_1241182_7_2_transradialfastV4_data.mat'
om_path = '/media/bugger/UBUNTU 20_0/data/v9_24012021_1241182_7_2_transradialfastV4_trajectory.mat'
image_path = '/media/bugger/UBUNTU 20_0/data/v9_24012021_1241182_7_2_transradialfastV4_25_image.mat'

A_img = scipy.io.loadmat(image_path)['one_slice']
A = scipy.io.loadmat(data_path)['data_array'][:, :, 0, 0, 0, 0]
A_traj = scipy.io.loadmat(om_path)['trajectory_array'][:, :, 0, :2]

NufftObj = NUFFT_cpu()

Nd = (260, 260)  # image size
Kd = (520, 520)  # k-space size
Jd = (4, 4)  # interpolation size

# om is the trajectory...
NufftObj.plan(A_traj.reshape(-1, 2), Nd, Kd, Jd)

# kspace is... the loaded mat file
# I dont see any density compensation
image0 = NufftObj.solve(A.reshape(-1, 1), solver='cg', maxiter=20)
hplotf.plot_3d_list(image0, augm='np.abs')
hplotf.plot_3d_list(image0, augm='np.abs')



from pynufft import NUFFT
import numpy
A = NUFFT()
om = numpy.random.randn(10,2)
Nd = (64,64)
Kd = (128,128)
Jd = (6,6)
A.plan(om, Nd, Kd, Jd)
x=numpy.random.randn(*Nd)
y = A.forward(x)
hplotf.plot_3d_list([x, y], augm='np.abs')