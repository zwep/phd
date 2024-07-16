"""
Small experiemtn to check how long certain lines are
"""

import skimage.data as data
import cv2
import numpy as np
import helper.plot_fun as hplotf
import helper.array_transf as harray
A = np.mean(data.astronaut(), axis=-1)
A = harray.scale_minmax(A)
plt.imshow(data.astronaut())
for i in [1, 10, 100, 200]:
    pred_canny = cv2.Canny((A * 255).astype(np.uint8), i, 500)
    hplotf.plot_3d_list(pred_canny)
index_to_go = np.ones(pred_canny.shape)
line_collection = []
n_x, n_y = pred_canny.shape
j = 79
i = 86
for i in range(n_x):
    for j in range(n_y):
        sel_pixel = pred_canny[i, j]
        index_status = index_to_go[i, j]
        if index_status == 1:
            if sel_pixel > 0:
                temp_line = []
                temp_line.append([i, j])
                index_to_go[i, j] = 0
                # Remember starting position
                base_i = i
                base_j = j

                neighbours = pred_canny[max(i-1, 0):min(i+2, n_x), max(j-1, 0):min(j+2, ny)]
                status_neighbours = index_to_go[max(i-1, 0):min(i+2, n_x), max(j-1, 0):min(j+2, ny)]
                base_neighbours_index = np.argwhere(neighbours * status_neighbours > 0)

                for i_point in base_neighbours_index:

                    neighbours = pred_canny[max(i-1, 0):min(i+2, n_x), max(j-1, 0):min(j+2, ny)]
                    status_neighbours = index_to_go[max(i-1, 0):min(i+2, n_x), max(j-1, 0):min(j+2, ny)]
                    base_neighbours_index = np.argwhere(neighbours * status_neighbours > 0)


                # Nu iteratief proces in...

            else:
                continue
        else:
            continue

input_canny = cv2.Canny((inputt*255).astype(np.uint8), 128, 196)
hplotf.plot_3d_list(input_canny)

perc_canny_increase = (pred_canny - input_canny) / input_canny