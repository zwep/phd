import h5py
import numpy as np
import helper.plot_class as hplotc
import helper.misc as hmisc


ddata = '/home/bugger/Documents/data/pinn_fdtd/Phantom_1 (Dipole 0deg).mat'
mat_obj = hmisc.load_array(ddata)

reshape_size = tuple(mat_obj['Grid_size'][0][::-1])

import collections
sigma = mat_obj['sigma'].reshape(reshape_size)
eps = mat_obj['eps'].reshape(reshape_size)
collections.Counter(sigma[25].ravel())
collections.Counter(eps[25].ravel())
hplotc.SlidingPlot(sigma)
hplotc.ListPlot([eps[25], sigma[25]])

E = mat_obj['Efield'].reshape(reshape_size + (3,))
B = mat_obj['Bfield'].reshape(reshape_size + (3,))
D = mat_obj['Dfield'].reshape(reshape_size + (3,))
rho = mat_obj['rho'].reshape(reshape_size)
n_slice = E.shape[0]
sel_slice = n_slice//2

E_sel = E[sel_slice]
B_sel = B[sel_slice]
D_sel = D[sel_slice]
# THis shows that E_x and E_y are zero
hplotc.ListPlot([E_sel[:, :, 0], E_sel[:, :, 1], E_sel[:, :, 2]], augm='np.abs')
# THis shows that B_z is zero
hplotc.ListPlot([B_sel[:, :, 0], B_sel[:, :, 1], B_sel[:, :, 2]], augm='np.abs')

# Select the non-zero components
B_x_cpx = B_sel[:, :, 0]
B_y_cpx = B_sel[:, :, 1]
E_z_cpx = E_sel[:, :, 2]
D_z_cpx = D_sel[:, :, 2]

# Take the real.. or abs..?
# I dont know..
B_x = np.real(B_x_cpx)
B_y = np.real(B_y_cpx)
E_z = np.real(E_z_cpx)
D_z = np.real(D_z_cpx)

np.gradient(D_z)
# This shows that d_x(E_z) and d_y(E_z) are not zero!
# (not that strange though I guess)
# But I dont know which equation to use right now

# Although Im not sure about the axis... although it doesnt really matter
# I just state that axis=0 == the x-axis
# And that axis=1 == the y-axis
E_z__y = np.gradient(E_z, axis=1)
E_z__x = -np.gradient(E_z, axis=0)
hplotc.ListPlot([np.abs(E_z__y), np.abs(E_z__x)], cbar=True, vmin=(0,5), title='abs d_y(E_z) and d_x(E_z)')

# Also d_x(B_x) + d_y(B_y) is not zero
B_x__x = np.gradient(B_x, axis=0)
B_y__y = np.gradient(B_y, axis=1)
hplotc.ListPlot([np.abs(B_x__x + B_y__y)], cbar=True, vmin=(0, 6*np.abs(B_y__y.mean())), title='abs(d_x(B_x) + d_y(B_y))')

# What about Faradays Law?
# Nabla cross E = -d_t(B)
# I guess that in this case -d_t(B) = -omega B, where omega is the Lamor Freq
# THis should display the first components of the above equation
hplotc.ListPlot([B_x, E_z__y], augm='np.abs', cbar=True, vmin=[(0, 1e-7), (0, 5)], title='B_x and d_y(E_z)')
