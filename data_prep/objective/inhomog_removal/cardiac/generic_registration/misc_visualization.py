import os
import helper.misc as hmisc
import helper.plot_class as hplotc

"""
Here we plot and store B1 distributions
"""


def plot_and_store(file_path, dest_dir):
    base_name = hmisc.get_base_name(file_path)
    dest_path = os.path.join(dest_dir, base_name)
    loaded_array = hmisc.load_array(file_path)
    print('Shape of loaded array ', loaded_array.shape)
    fig_obj = hplotc.ListPlot(loaded_array, augm='np.abs', col_row=(2,4))
    fig_obj.figure.savefig(dest_path, bbox_inches='tight')



ddata = '/data/cmr7t3t/b1_distr'
slice_dir = ['4ch', 'p2ch', 'sa']
subdir = ['b1_plus', 'b1_minus']

for i_slice_dir in slice_dir:
    for i_subdir in subdir:
        dsource = os.path.join(ddata, i_slice_dir, i_subdir)
        dtarget = os.path.join(ddata, i_slice_dir, i_subdir + "_PNG")
        if not os.path.isdir(dtarget):
            os.makedirs(dtarget)
        for i_file in os.listdir(dsource):
            plot_and_store(os.path.join(dsource, i_file), dtarget)
            hplotc.close_all()


"""
Here we visualize the old and new B1 distributios from the slicing thing
 
"""
import os
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
import numpy as np
hplotc.close_all()
ddata = '/media/bugger/MyBook/data/simulated/cardiac/bart/sa'
ddata_old = '/media/bugger/MyBook/data/simulated/cardiac/bart/sa_old'
subdir = ['b1_plus', 'rho']
file_names = [f"V{i}.npy" for i in range(1, 15)]
for i_file in file_names:
    for isubdir in subdir:
        if 'rho' in isubdir:
            sel_data = os.path.join(ddata, 'sigma', i_file)
        else:
            sel_data = os.path.join(ddata, isubdir, i_file)
        sel_data_array = hmisc.load_array(sel_data)
        sel_data_old = os.path.join(ddata_old, isubdir, i_file)
        sel_data_array_old = hmisc.load_array(sel_data_old)
        print(sel_data_array.shape)
        if sel_data_array.ndim == 3:
            hplotc.ListPlot([np.abs(sel_data_array).sum(axis=0),
                             np.abs(sel_data_array_old).sum(axis=0)], title=f"{i_file} + {isubdir}")
        else:
            hplotc.ListPlot([np.abs(sel_data_array),
                             np.abs(sel_data_array_old)], title=f"{i_file} + {isubdir}")


"""
Visualize old and new data from b1 stuff
"""


import os
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
import numpy as np
hplotc.close_all()
ddata = '/data/cmr7t3t/biasfield_sa_mm1_B/train/input'
ddata_old = '/data/cmr7t3t/biasfield_sa_mm1_B_old/train/input'
file_names = os.listdir(ddata)[:10]
hplotc.close_all()
for i_file in file_names:
    sel_data = os.path.join(ddata, i_file)
    sel_data_old = os.path.join(ddata_old, i_file)
    sel_data_array = hmisc.load_array(sel_data)
    sel_data_array = sel_data_array[0] + 1j * sel_data_array[1]
    sel_data_array_old = hmisc.load_array(sel_data_old)
    sel_data_array_old = sel_data_array_old[0] + 1j * sel_data_array_old[1]
    print(sel_data_array.shape)
    if sel_data_array.ndim == 3:
        fig_obj = hplotc.ListPlot([np.abs(sel_data_array).sum(axis=0),
                         np.abs(sel_data_array_old).sum(axis=0)], title=f"{i_file}")
    else:
        fig_obj = hplotc.ListPlot([np.abs(sel_data_array),
                         np.abs(sel_data_array_old)], title=f"{i_file} ")
    fig_obj.figure.savefig(os.path.join('/data/seb', hmisc.get_base_name(i_file) + '.png'))