import h5py
import numpy as np
import helper.plot_class as hplotc
import helper.misc as hmisc
import helper.array_transf as harray
import os

ddata = '/nfs/arch11/researchData/USER/emeliado/PINN_FDTD_with_Seb/MatLab_Dataset_PINN_FDTD'

file_list = os.listdir(ddata)
file_list = [x for x in file_list if x.endswith('mat')]
sel_file = file_list[0]
sel_file_path = os.path.join(ddata, sel_file)
#
loaded_array = hmisc.load_array(sel_file_path)
for k, v in loaded_array.items():
    if not k.startswith('__'):
        print(k, v.shape)

reshape_size = tuple(loaded_array['Grid_size'][0][::-1])
n_slice = reshape_size[0]
sel_slice = n_slice//2
#
XAxis = loaded_array['XAxis'].ravel()
x = (XAxis[-1] + XAxis[1:])/2
YAxis = loaded_array['YAxis'].ravel()
y = (YAxis[-1] + YAxis[1:])/2

# Define the 'constant' arrays...
rho_array = loaded_array['rho'].reshape(reshape_size)
# Conductivity sigma
sigma_array = loaded_array['sigma'].reshape(reshape_size)
# Permitivity epsilon
eps_array = loaded_array['eps'].reshape(reshape_size)
# Permeability mu...?
# Reshape the B and E fields
B_array = loaded_array['Bfield'].reshape(reshape_size + (3,))
E_array = loaded_array['Efield'].reshape(reshape_size + (3,))
D_array = loaded_array['Dfield'].reshape(reshape_size + (3,))
# Prepare arrays for display
B_array_disp = np.abs(B_array[sel_slice])
B_array_disp = harray.scale_minmax(B_array_disp)
E_array_disp = np.abs(E_array[sel_slice])
E_array_disp = harray.scale_minmax(E_array_disp)
D_array_disp = np.abs(D_array[sel_slice])
D_array_disp = harray.scale_minmax(D_array_disp)
# Obtain the index of the location of the coil..
y_ind, x_ind, z_ind = np.unravel_index(B_array_disp.argmax(), B_array_disp.shape)
 
# Plot the E and B fields in rgb
plot_obj = hplotc.ListPlot([B_array_disp[None], E_array_disp[None], D_array_disp[None]], cmap='rgb', sub_col_row=(1,1), vmin=(0, 0.00004), cbar=True)
plot_obj.figure.savefig('/local_scratch/sharreve/E_B_D_fields.png')

# Plot the E and B fields in x, y , z
plot_obj = hplotc.ListPlot([np.moveaxis(B_array_disp, -1, 0),
                            np.moveaxis(E_array_disp, -1, 0),
                            np.moveaxis(D_array_disp, -1, 0)],
                           vmin=(0, B_array_disp[:, :, 0].mean() * 3))
plot_obj.figure.savefig('/local_scratch/sharreve/E_B_D_field_x_y_z.png')

# Plot the rho/sigma/eps array
plot_obj = hplotc.ListPlot([rho_array[sel_slice], sigma_array[sel_slice], eps_array[sel_slice]], cbar=True)
plot_obj.figure.savefig('/local_scratch/sharreve/rho_sigma_eps_fields.png')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2)
ax[0].hist(sigma_array.ravel(), label='sigma')
ax[0].legend()
ax[1].hist(eps_array.ravel(), label='eps')
ax[1].legend()
fig.savefig('/local_scratch/sharreve/eps_sigma_hist.png')

plot_obj = hplotc.ListPlot([rho_array[sel_slice], sigma_array[sel_slice], eps_array[sel_slice]], cbar=True)
plot_obj.figure.savefig('/local_scratch/sharreve/rho_sigma_eps_fields.png')



# Check whether the Dfield is really equal to Eps * Efield..
Dfield_approx = np.abs(eps_array[sel_slice][:, :, None] * E_array[sel_slice])
D_array_disp = np.abs(D_array[sel_slice])

point_of_interest = [25, 105, 105]
E_array[sel_slice][0, 0, 0] * eps_array[sel_slice][0, 0]
eps_array[sel_slice][0, 0]
d_value = D_array[25, 105, 105, 0]
e_value = E_array[25, 105, 105, 0]
eps_value = eps_array[25, 105, 105]
print(f"D_x value at (25, 105, 105): {d_value}")
print(f"E_x value at (25, 105, 105): {e_value}")
print(f"D_x/E_x value at (25 105, 105): {np.abs(d_value)/np.abs(e_value)}")
print(f"Eps value at (25 105, 105): {eps_value}")

plot_obj = hplotc.ListPlot([np.abs(D_array[sel_slice, :, :, 0]),
                            np.abs(Dfield_approx[:, :, 0])], cbar=True, subtitle=[['Dfield'], ['Dfield approx']])

plot_obj.figure.savefig('/local_scratch/sharreve/D_fields.png')

"""
Check Smash files
"""
dsmash = '/nfs/arch11/researchData/USER/emeliado/PINN_FDTD_with_Seb/Simulations/Test_15.smash_Results'
file_input = '0aac2910-4c82-4440-bfcd-1050bd9fc460_Input.h5'
file_output = '0aac2910-4c82-4440-bfcd-1050bd9fc460_Output.h5'
with h5py.File(os.path.join(dsmash, file_input), 'r') as f:
    print(f.keys())

# Scroll through some keys in the output file.
# Found nothing interesting
with h5py.File(os.path.join(dsmash, file_output), 'r') as f:
    print(f['AllMaterialMaps']['fbdc0b32-e76e-4945-9aab-7dc995420e4c']['159b49de-e92e-435a-a5ac-94391009b5a4']['Property_0']['_ClassInfo'])
    hmisc.print_dict(f)

# Check which names are there
file_paths = [os.path.join(dsmash, x) for x in os.listdir(dsmash) if x.endswith('Input.log')]
list_of_names = []
for sel_file_log in  file_paths:
    with open(sel_file_log, 'r') as f:
        A = f.readlines()
    import re
    single_line = [x for x in A if x.startswith('[INFO]: Simulation')][0]
    simulation_name = re.findall('.*Simulation (.*) started.*', single_line)[0]
    # print(simulation_name)
    list_of_names.append(simulation_name)

def sort_fun(x):
    _, _, dipole_deg = x.split()
    degree = re.findall('([0-9]+)deg', dipole_deg)[0]
    return int(degree)

sorted(list_of_names, key=lambda x: sort_fun(x))