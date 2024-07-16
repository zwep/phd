"""
Stupid functions to generate some data...
"""

import numpy as np
import helper.array_transf as harray


def get_elipse(n_x, n_y, amp_factor=1/3):
    # n_x n_y is the size of the image
    # amp_factor scales the amplitude of the ellipse

    n_points = 999
    theta_range = np.linspace(-np.pi, np.pi, n_points)
    x = amp_factor * n_x * np.cos(theta_range) + n_x / 2
    y = amp_factor * n_y * np.sin(theta_range) + n_y / 2
    A = np.zeros((n_x, n_y))
    A[x.astype(int), y.astype(int)] = 1
    # Such an overkill.. but it does the job
    A_convex = harray.convex_hull_image(A)
    return A_convex


def get_gaussian_blobs(n_x, n_y, n_c):
    x = np.linspace(-2, 2, n_x)
    y = np.linspace(-2, 2, n_y)
    X, Y = np.meshgrid(y, x)
    # Badly programmed. But it works
    circle_blobs = []
    # We wanted to place the blobs around a circle..
    # That is why I am taking n_c places on a 0..2pi range
    theta_range = np.linspace(0, 2 * np.pi, n_c + 1)[:-1] + np.pi/8
    for i_theta in theta_range:
        x_offset = 2 * np.cos(i_theta)
        y_offset = 2 * np.sin(i_theta)
        temp = np.exp(-((X + x_offset) ** 2 + (Y + y_offset) ** 2) / 2)
        circle_blobs.append(temp)
    Z = np.stack(circle_blobs, axis=0)
    return Z


def get_sentivitiy_maps(n_x, n_y, n_c):
    return get_gaussian_blobs(n_x, n_y, n_c) * get_elipse(n_x, n_y)

