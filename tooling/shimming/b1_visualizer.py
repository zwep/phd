import helper.plot_class as hplotc
import numpy as np
from objective_helper.fourteenT import ReadMatData

ddata = '/home/bugger/Documents/data/14T/7T'
sel_mat_file = '8 Channel Dipole Array 7T_ProcessedData.mat'
mat_reader = ReadMatData(ddata=ddata, mat_file=sel_mat_file)
b1_obj = mat_reader.read_B1_object()
b1p_array = b1_obj['b1p']
n_c, n_z, _, _ = b1p_array.shape
b1p_slice = b1p_array[:, n_z//2]

def get_max_xy(x_array):
    if x_array.ndim == 2:
        x_array = x_array[None]
    elif x_array.ndim > 4:
        print('Too much dimensions')
        return []

    x_y_list = [np.unravel_index(np.abs(x).argmax(), x_array.shape[-2:]) for x in x_array]
    return x_y_list

x_y_list = get_max_xy(b1p_slice)
import matplotlib.pyplot as plt

random_shim_list = [np.random.uniform(0, 2 * np.pi, n_c) for _ in range(10)]
fig, ax = plt.subplots()
ax_imshow = ax.imshow(np.zeros(b1p_array.shape[-2:]))
for i_shim in random_shim_list:
    b1p_shim = (b1p_slice.T @ i_shim).T
    ax_imshow.set_data(np.abs(b1p_shim))
    ax_imshow.set_clim(0, np.abs(b1p_shim).max())
    fig.canvas.draw()
    plt.pause(0.1)

    for ii, (ix, iy) in enumerate(x_y_list):

    ax[ii].imshow(np.abs(b1p_slice[ii]))



[x.argmax() for x in b1p_slice]
hplotc.ListPlot(, augm='np.abs')