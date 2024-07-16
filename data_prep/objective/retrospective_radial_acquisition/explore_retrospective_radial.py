import numpy as np
import sigpy.mri
import matplotlib.pyplot as plt
import scipy.io
import helper.plot_class as hplotc

"""
This is the data from Matthijs

The data is aqcquired properly. But forgot to set proper shim settings...
"""

ddata_not_sorted = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_11_24/recon_data.mat'

non_sorted_array = scipy.io.loadmat(ddata_not_sorted)['recon_data']
non_sorted_data_array = non_sorted_array[0][0]
non_sorted_data_array.shape
non_sorted_noise_array = non_sorted_array[4][0]

hplotc.SlidingPlot(non_sorted_data_array[:, ::24])

ddata_sorted = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_11_24/recon_sorted.mat'
sorted_array = scipy.io.loadmat(ddata_sorted)['recon_sorted']
sorted_array[0][0].shape



# This is how the data should have looked like
ddata_unsorted = '/media/bugger/MyBook/data/7T_data/cardiac_unsorted_data/V9_19531/unsorted_data/v9_02052021_1431531_6_2_transradialfastV4.mat'
zz = scipy.io.loadmat(ddata_unsorted)['reconstructed_data']
hplotc.SlidingPlot(zz[:, ::24])

"""
Explore retrospective radial for fantom...
"""

ddata = '/media/bugger/MyBook/data/7T_scan/phantom/2021_10_20/fantom_data.mat'
fantom_unsorted = scipy.io.loadmat(ddata)
n_points = fantom_unsorted['unsorted_data'].shape[0]
n_spokes = 25
width = 4
ovs = 1.25
n_coils = 8
n_cardiac_phases = 10
img_shape = (528, 528)
one_phase_data = fantom_unsorted['unsorted_data'][:, :n_spokes * n_coils]
temp = one_phase_data.reshape((img_shape[0], n_spokes, n_coils))
temp = np.moveaxis(temp, -1, 0)

trajectory_radial = sigpy.mri.radial(coord_shape=(n_spokes, n_points, 2), img_shape=img_shape)
fig, ax = plt.subplots()
ax.set_ylim(-img_shape[0]//2, img_shape[0]//2+1)
ax.set_xlim(-img_shape[0]//2, img_shape[0]//2+1)
ax.scatter(trajectory_radial[0, :, 0], trajectory_radial[0, :, 1])
ax.scatter(trajectory_radial[1, :, 0], trajectory_radial[1, :, 1])
ax.scatter(trajectory_radial[10, :, 0], trajectory_radial[10, :, 1])
# #
vec1 = [x_coords[0], y_coords[0]]
vec2 = [0, -1]
calc_angle2 = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
# #
# I believe with degree 0: x = 0, y = -kmax/2 .. kmax/2 - 1
trajectory_radial = trajectory_radial.reshape(-1, 2)
dcf = np.sqrt(trajectory_radial[:, 0] ** 2 + trajectory_radial[:, 1] ** 2)

res_gridded = []
for i_coil in temp:
    # input = i_coil.reshape(-1) * dcf
    # coord = trajectory_radial
    # oversamp = 1.25
    width = 4
    # oshape = img_shape
    temp_img = sigpy.nufft_adjoint(i_coil.reshape(-1) * dcf, coord=trajectory_radial, oshape=img_shape, width=width, oversamp=ovs)
    res_gridded.append(temp_img)

res_gridded = np.array(res_gridded)
hplotc.ListPlot(res_gridded.sum(axis=0), augm='np.abs')
##


# 8 coils..
# 10 phases
# Hoeveel spokes..?
# Geen ideeeee....

# RC_retro_total_nr_intp_profiles = `RC_total_nr_profiles * `UGN8_CARD_rec_phases / `UGN9_CARD_acq_phases
# 2000                            = 250                   *  x / 10  --> x = 80
# 2000                            = 250                   *  10 / x  --> x = 1.25  # This is the oversample factor..

ddata_sorted = '/media/bugger/MyBook/data/7T_scan/phantom/2021_10_20/se_26726'


import reconstruction.ReadListData as readlist
dlist = '/media/bugger/MyBook/data/7T_scan/cardiac/20180926_scan/raw_10_seb_trigger.list'
list_obj = readlist.DataListImage(dlist)
imgarray = list_obj.get_image_data()
hplotc.SlidingPlot(imgarray[0])


import reconstruction.ReadCpx as read_cpx
dcpx = '/media/bugger/MyBook/data/7T_scan/cardiac/20190814_raw/V9_998/as_14082019_1731410_25_3_b1shimseriesV4.cpx'
cpx_obj = read_cpx.ReadCpx(dcpx)
cpx_img = cpx_obj.get_cpx_img()
hplotc.SlidingPlot(np.rot90(np.squeeze(np.abs(cpx_img)).sum(axis=0).sum(axis=0), axes=(-2, -1)))


"""
Get the sorted data and see how many spokes it expected...
"""

"""
Fantom data 9 december
"""
ddata_sorted = "/media/bugger/WORK_USB/2021_12_09/mat_data"
# DEZE HEB IK NIET GESORTEERD....
# Net ff remote gedaan... ik kom uit op 28 spokes.
# Dit is van de file met 100 sampling...
# Hier klopt iets toch niet......

# Enn dan 14 spokes met dedegene met 10% sampling.
# Hier klopt gewoon iets niet....


"""
Bart data 1  december
"""
ddata_sorted = "/media/bugger/WORK_USB/bart_data/bart_17_2_sorted.mat"
bart_array = scipy.io.loadmat(ddata_sorted)['bart_17_2_sorted']
# Looks like 8 spokes.......?
print(bart_array.shape)
"""
Matthijs data 24 nov
"""

ddata_sorted = "/media/bugger/MyBook/data/7T_scan/cardiac/2021_11_24/recon_sorted.mat"
matthijs_array = scipy.io.loadmat(ddata_sorted)['recon_sorted'][0][0]
# Looks like 16 spokes.......?
print(matthijs_array.shape)


"""
Unsorted untriggered stuff
"""

import os
import numpy as np
import scipy.io
import reconstruction.ReadCpx as read_cpx
import matplotlib.pyplot as plt
# This is how the data should have looked like
ddata_unsorted = '/media/bugger/MyBook/data/7T_data/cardiac_unsorted_data/V9_19531'
ddata_unsorted_scanned = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac/V9_19531'
ddata_notrig_mat = os.path.join(ddata_unsorted, 'v9_02052021_1434042_8_2_transradial_no_trigV4.mat')
ddata_notrig_cpx = os.path.join(ddata_unsorted_scanned, 'v9_02052021_1434042_8_2_transradial_no_trigV4.mat')

import helper.misc as hmisc
import helper.plot_class as hplotc
mat_obj = hmisc.load_array(ddata_notrig_mat, data_key='unsorted_data')[0][0]
mat_obj_sorted = hmisc.load_array(ddata_notrig_cpx, data_key='reconstructed_data')
mat_obj_sorted.shape
mat_obj[0][0].shape
hplotc.SlidingPlot(mat_obj_sorted.T)

# The .sin file was not exported.... in the .par file
n_coil = 24
n_dyn = 80
n_spokes = 66
center_index = mat_obj.shape[0]//2

coil_signal = []
for i_coil in range(n_coil):
    coil_info = mat_obj[center_index, i_coil::n_coil]
    coil_signal.append(coil_info)

coil_signal = np.array(coil_signal)
n_acquired_spokes = coil_signal.shape[1]

TR = 4.5 * 10 ** -3
TR = 29.28 / n_acquired_spokes # 29.28 is the total scan duration

time_range = np.arange(n_acquired_spokes) * TR
fig, ax = plt.subplots(8)
for i_spoke in range(0, n_spokes, 10):
    for ii, sel_coil in enumerate(coil_signal[-8:]):
        # ax[ii].plot(time_range, np.abs(sel_coil))
        ax[ii].plot(time_range[i_spoke::n_spokes], np.abs(sel_coil)[i_spoke::n_spokes], 'r-o', alpha=0.2)

# Did I just found something for the breathing? Lets check back at home if it relates to the images frmo the recon
hplotc.SlidingPlot(mat_obj_sorted.T)
test_array = np.squeeze(dyn_recon_array.sum(axis=0))[:, ::-1, ::-1]
probable_breathing_coil_index = 1
cluster_spoke_array = []
for i_spoke in range(0, n_spokes, 1):
    temp = np.abs(coil_signal[-probable_breathing_coil_index])[i_spoke::n_spokes]
    cluster_spoke_array.append(temp)

cluster_spoke_array = np.array(cluster_spoke_array)
peak_spokes = np.argsort(cluster_spoke_array[:, 20])[-10:]
lines_that_short_breathing = cluster_spoke_array[peak_spokes]
timing_of_spokes = []
for i_spoke in peak_spokes:
    timing_of_spokes.append(time_range[i_spoke::n_spokes])

timing_of_spokes = np.array(timing_of_spokes)

fig, ax = plt.subplots(2, figsize=(30, 20))
test_array = np.squeeze(mat_obj_sorted.T)
aximshow = ax[0].imshow(np.abs(test_array[0]))
for it, iy in zip(timing_of_spokes, lines_that_short_breathing):
    ax[1].plot(it, iy)

ax[0].set_title('Free breathing dynamic radial acquisition')
ax[1].set_title('Intensity of the centre of a fixed spoke angle over the dynamics')
for _ in range(5):
    aximshow = ax[0].imshow(np.abs(test_array[0]))
    ax[0].set_axis_off()

    for it, iy in zip(timing_of_spokes, lines_that_short_breathing):
        ax[1].plot(it, iy)

    for i_time in range(n_dyn):
        aximshow.set_data(np.abs(test_array[i_time]))
        plt.pause(0.1)
        if i_time > 0:
            ax[1].collections[0].remove()
        ax[1].scatter(timing_of_spokes[:, i_time], lines_that_short_breathing[:, i_time], c='k')

    ax[0].clear()
    ax[1].clear()
    ax[0].set_title('Free breathing dynamic radial acquisition')
    ax[1].set_title('Intensity of the centre of a fixed spoke angle over the dynamics')



# Lets try some fastICA...?
import sklearn.decomposition
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
X, _ = load_digits(return_X_y=True)
X.shape
transformer = FastICA(n_components=7, random_state=0)
X_transformed = transformer.fit_transform(X)
X_transformed.shape


# Sooo...?

transformer = FastICA(n_components=10, random_state=0, whiten=True)
X_transformed = transformer.fit_transform(np.abs(coil_signal).T)
fig, ax = plt.subplots(2)
ax[0].plot(time_range, X_transformed[:, 0])
ax[1].plot(time_range, X_transformed[:, 1])
fig, ax = plt.subplots()
for i_ica in X_transformed.T:
    result_fft = np.abs(np.fft.fft(i_ica))
    result_fft_freq = np.fft.fftfreq(len(i_ica), d=TR)
    ax.plot(result_fft_freq, result_fft)
