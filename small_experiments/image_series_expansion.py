"""
Maybe expanding in some basis.. approximating each basis can help with stuff
Or maybe even different resolutions


source: https://stackoverflow.com/questions/4258106/how-to-calculate-a-fourier-series-in-numpy
"""

import helper.array_transf as harray
import helper.plot_class as hplotc
import matplotlib.pyplot as plt
import numpy as np
import scipy


"""
Approximate a 1-D function with a disjoint set of basis functions
"""

time = np.linspace(-2, 2, 100)
period = 2
y = np.sinc(time) ** 2

y_fft = np.fft.fftshift(np.fft.fft(y))
N = len(y_fft)
fig, ax = plt.subplots()
ax.plot(y, 'r', alpha=0.5)
n_bins = 20
length_bin = N // n_bins
res_list = []
for i_bin in range(0, n_bins//2):
    print(i_bin)
    res = np.zeros(N, dtype=np.complex64)
    if i_bin == 0:
        min_coord = N // 2 - length_bin // 2
        max_coord = N // 2 + length_bin // 2
        res[min_coord:max_coord] = y_fft[min_coord:max_coord]
    else:
        min_coord_left = N // 2 - length_bin // 2 - i_bin * length_bin
        min_coord_left = max(min_coord_left, 0)
        max_coord_left = min_coord_left + length_bin
        res[min_coord_left:max_coord_left] = y_fft[min_coord_left: max_coord_left]

        min_coord_right = N // 2 + length_bin // 2 + (i_bin-1) * length_bin
        max_coord_right = min_coord_right + length_bin
        max_coord_right = min(max_coord_right, N)

        res[min_coord_right: max_coord_right] = y_fft[min_coord_right: max_coord_right]
    res_list.append(res)


y_approx_list = [np.fft.ifft(np.fft.ifftshift(x)) for x in res_list]
[plt.plot(x, label=f'approximation {i}') for i, x in enumerate(y_approx_list)]
plt.plot(np.sum(y_approx_list, axis=0), 'k--', label='sum approximation')
plt.plot(y, 'r.', label='target')
plt.title('Overview approximations')
plt.legend()

"""
Did it for a single signal..
Now try an image.. using SQUARE cut outs
"""

import skimage.data
import scipy.interpolate
import scipy.optimize

A_img = skimage.data.astronaut().sum(axis=-1)
A = np.fft.fftshift(np.fft.fftn(A_img, axes=(-2, -1)))

# Now do 2D
N = len(A)
n_bins = 20
length_bin = N // n_bins
res_list = []
prev_coord = None
for i_bin in range(0, n_bins//2):
    print(i_bin)
    print(prev_coord)
    res = np.zeros((N, N), dtype=np.complex64)
    if i_bin == 0:
        min_coord = N // 2 - length_bin // 2
        max_coord = N // 2 + length_bin // 2
        res[min_coord:max_coord, min_coord:max_coord] = A[min_coord:max_coord, min_coord:max_coord]
        prev_coord = (min_coord, max_coord)
        print(min_coord, max_coord)
    else:
        min_coord_left = N // 2 - length_bin // 2 - i_bin * length_bin
        min_coord_left = max(min_coord_left, 0)
        max_coord_right = N // 2 + length_bin // 2 + i_bin * length_bin
        max_coord_right = min(max_coord_right, N)

        res[min_coord_left:max_coord_right, min_coord_left:max_coord_right] = A[min_coord_left:max_coord_right, min_coord_left:max_coord_right]
        res[prev_coord[0]: prev_coord[1], prev_coord[0]: prev_coord[1]] = 0
        prev_coord = (min_coord_left, max_coord_right)
        print(min_coord_left, max_coord_right)

    res_list.append(res)

hplotc.SlidingPlot(np.array(res_list))

A_approx_list = [np.fft.ifftn(np.fft.ifftshift(x), axes=(-2, -1)) for x in res_list]
hplotc.SlidingPlot(np.array(A_approx_list))

fig, ax = plt.subplots(len(A_approx_list))
for i, i_ax in enumerate(ax):
    # i_ax.imshow(np.abs(np.sum(A_approx_list[0:(i+1)], axis=0)))
    plt.figure()
    plt.imshow(np.abs(np.sum(A_approx_list[0:(i+1)], axis=0)))

"""
same SQUARE cut outs, but with non linear cuts
"""

import skimage.data
import scipy.interpolate
import scipy.optimize

A_img = skimage.data.astronaut().sum(axis=-1)
A = np.fft.fftshift(np.fft.fftn(A_img, axes=(-2, -1)))

# Estimate offsets in a non-linear manner
x_index = np.arange(0, 256)
x_target = np.abs(A[256, 0:256])
res = scipy.optimize.curve_fit(lambda t, a, b, c: a * np.exp(b*t) + c * t,  x_index,  x_target, p0=(4, 0.1, 0.1))
res_fun = res[0][0] * np.exp(res[0][1] * x_index) + res[0][2] * x_index
cut_off_points = 256 - harray.split_area(res_fun, 10)[::-1]

# Visualize the new estimation
plt.plot(x_target[::-1])
plt.plot(res_fun[::-1])
for i_line in cut_off_points:
    plt.vlines(x=i_line, ymin=0, ymax=np.max(res_fun))

# Now do 2D
n_bins = len(cut_off_points)
res_list = []
prev_coord = None
for i_bin in range(0, n_bins):
    i_offset = cut_off_points[i_bin]
    print(i_bin, i_offset)
    print(prev_coord)
    res = np.zeros((N, N), dtype=np.complex64)
    min_coord = N // 2 - i_offset + 1
    min_coord = max(min_coord, 0)
    max_coord = N // 2 + i_offset + 1
    max_coord = min(max_coord, N)
    # print(min_coord, max_coord)
    res[min_coord:max_coord, min_coord:max_coord] = A[min_coord:max_coord, min_coord:max_coord]
    if prev_coord is not None:
        res[prev_coord[0]: prev_coord[1], prev_coord[0]: prev_coord[1]] = 0

    prev_coord = (min_coord, max_coord)
    print(prev_coord)
    res_list.append(res)

hplotc.SlidingPlot(np.array(res_list))

A_approx_list = [np.fft.ifftn(np.fft.ifftshift(x), axes=(-2, -1)) for x in res_list[2:]]
hplotc.SlidingPlot(np.array(A_approx_list))

A_img_list = [np.abs(np.sum(A_approx_list[0:(i+1)], axis=0)) for i in range(len(A_approx_list))]
hplotc.SlidingPlot(np.array(A_img_list))

"""
Circular cut outs with NON linear distance and non square image
"""

import skimage.data
import scipy.interpolate
import scipy.optimize

A_img = skimage.data.astronaut().sum(axis=-1)
A_img = A_img[:364, :499]
plt.imshow(A_img)
A = np.fft.fftshift(np.fft.fftn(A_img, axes=(-2, -1)))
plt.imshow(np.log(np.abs(A)))

n_y, n_x = A_img.shape
x_range = np.linspace(-n_x//2, n_x//2, n_x+1)[:-1]
y_range = np.linspace(-n_y//2, n_y//2, n_y+1)[:-1]
X, Y = np.meshgrid(x_range, y_range)

# Now do 2D
for n_bins in range(6,20):
    n_bins = 5
    full_circle = np.sqrt((X / (0.5 * n_x)) ** 2 + (Y / (0.5 * n_y)) ** 2)

    # Estimate offsets in a non-linear manner with the x-axis....
    x_index = np.arange(0, n_x//2)
    x_target = np.abs(A[n_y//2, :n_x//2])
    coeff_obj = scipy.optimize.curve_fit(lambda t, a, b, c: a * np.exp(b*t) + c * t,  x_index,  x_target, p0=(4, 0.1, 0.1))
    coeff_x = coeff_obj[0]
    approx_fun_x = coeff_x[0] * np.exp(coeff_x[1] * x_index) + coeff_x[2] * x_index
    plt.plot(approx_fun_x)
    plt.plot(x_target)
    cut_off_points = harray.split_area(approx_fun_x[::-1], n_bins)
    cut_off_points[-1] = 9000
    print(n_bins, len(cut_off_points))

# Visualize these elliptical cutoff values...
plt.plot(x_target[::-1])
plt.plot(approx_fun_x[::-1])
for i_line in cut_off_points[:-1]:
    plt.vlines(x=i_line, ymin=0, ymax=np.max(approx_fun_x))

res_list = []
prev_radius = None
for i_radius in cut_off_points:
    i_radius = int(i_radius)
    if prev_radius is None:
        selection_circle = full_circle <= (i_radius / (n_x//2))
        res = A * selection_circle.astype(int)
    else:
        prev_selection_circle = full_circle <= (prev_radius / (n_x//2))
        selection_circle = full_circle <= (i_radius / (n_x//2))
        res = A * (selection_circle.astype(int) - prev_selection_circle.astype(int))

    prev_radius = i_radius
    res_list.append(res)

hplotc.SlidingPlot(np.array(res_list))

A_approx_list = [np.fft.ifftn(np.fft.ifftshift(x), axes=(-2, -1)) for x in res_list]
hplotc.SlidingPlot(np.array(A_approx_list))

A_img_list = [np.abs(np.sum(A_approx_list[0:(i+1)], axis=0)) for i in range(len(A_approx_list))]
hplotc.SlidingPlot(np.array(A_img_list))
A_img_list = [np.abs(np.sum(np.real(A_approx_list[0:(i+1)]), axis=0)) for i in range(len(A_approx_list))]
hplotc.SlidingPlot(np.array(A_img_list))

"""
Now try and do something with SVD>..
"""

u_svd, s_svd, vh_svd = np.linalg.svd(A_img, full_matrices=False)
N = len(A_img)
n_svd = 100
res_list = []
index_range = [(np.min(x), np.max(x)) for x in np.split(np.arange(0, 100), 10)]
for i_min, i_max in index_range:
    print(i_min, i_max)
    res = u_svd[:, i_min:i_max] @ np.diag(s_svd[i_min:i_max]) @ vh_svd[i_min:i_max, :]
    res_list.append(res)

hplotc.SlidingPlot(np.array(res_list))

fig, ax = plt.subplots(len(res_list))
for i, i_ax in enumerate(ax):
    plt.figure()
    plt.imshow(np.abs(np.sum(res_list[0:(i+1)], axis=0)))
