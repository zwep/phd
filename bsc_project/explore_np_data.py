
"""
I have send some numpy data. This is a simple script to load and inspect the data

Used abbreviations in this file

n_c - number of coils. This can vary from 8 - 24. The last 8 coils are only needed
n_x, n_y - dimension of the data. This can vary from something like 256 - 700 or so.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

dir_data = '/media/bugger/MyBook/data/7T_data/cardiac/shimseries'
# Filter out some files.. they were accidently added to the .zip
list_files = [x for x in os.listdir(dir_data) if ('radial' not in x) and (x.endswith('npy'))]

# Get one file
sel_index = 0
sel_file = list_files[sel_index]
dir_file = os.path.join(dir_data, sel_file)

# Load it...
A = np.load(dir_file)
print('Shape of the file is... ', A.shape)
if A.ndim == 3:
    print('Wrong dimensions.. we should skip this file..')
    print('Dimensions should be.. n_c, n_c, n_x, n_y')

# Select only the last 8. Needed because of how the scanner stores the data
A_coils = A[-8:, -8:]
n_c, n_c, n_x, n_y = A_coils.shape

fig, ax = plt.subplots(ncols=n_c, nrows=n_c)
ax = ax.ravel()
A_reshape = A_coils.reshape((n_c ** 2, n_x, n_y))
for i, i_image in enumerate(A_reshape):
    ax[i].imshow(np.abs(i_image), vmin=(0, 20000))

# Show only the top row
fig, ax = plt.subplots(ncols=n_c, nrows=1)
for i, i_image in enumerate(A_coils[0, :]):
    ax[i].imshow(np.abs(i_image).sum(axis=0), vmin=(0, 20000))
fig.suptitle('Display of receive sensitivity of coil 1 with respect to all other coils')


# Show only the first column
fig, ax = plt.subplots(ncols=1, nrows=n_c)
for i, i_image in enumerate(A_coils[:, 0]):
    ax[i].imshow(np.abs(i_image), vmin=(0, 20000))
fig.suptitle('Display of transmit sensitivity of coil 1 with respect to all other coils')


# Another way to look at the data is by summing
A_abs_summed = np.abs(A_coils).sum(axis=0).sum(axis=0)
fig, ax = plt.subplots()
ax.imshow(A_abs_summed)
fig.suptitle('No interference patterns')

# By taking the absolute value over a different axis.. we see different patterns
A_abs_ax_0 = np.abs(A_coils.sum(axis=0)).sum(axis=0)
fig, ax = plt.subplots()
ax.imshow(A_abs_ax_0)
fig.suptitle('Summed over first axis')

# By taking the absolute value over a different axis.. we see different patterns
A_abs_ax_1 = np.abs(A_coils.sum(axis=1)).sum(axis=0)
fig, ax = plt.subplots()
ax.imshow(A_abs_ax_1)
fig.suptitle('Summed over second axis')
