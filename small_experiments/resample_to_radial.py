# encoding: utf-8


"""
Small program to check how the radial sampling went..
"""

import numpy as np
import os
import importlib

import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
import helper.nufft_recon as hnufft
from pynufft import NUFFT_cpu


NufftObj = NUFFT_cpu()

Nd = (256, 256)  # image size
Kd = (512, 512)  # k-space size
Jd = (6, 6)  # interpolation size


import helper.phantom
# Load image
image = helper.phantom.phantom(Nd[0])

# Define om nr 2
a = 0  # offset
c = 1  # -1, 1, -2, 2
b = np.pi/(6*np.pi ** (1/c))  # scaling factor
theta_max = (np.pi - a) / b  # Fuck yeah, math
theta = np.linspace(0, theta_max, Kd[0])
x_spiral = np.cos(theta) * (a + b * theta ** (1/c))
y_spiral = np.sin(theta) * (a + b * theta ** (1/c))

# Define om nr 3
y_line = np.zeros(Kd[0])
x_line = np.linspace(-np.pi, np.pi, Kd[0])

single_spiral = np.vstack([x_spiral, y_spiral]).T
single_line = np.vstack([x_line + 0.5, y_line]).T

om_spiral = hnufft.get_rot_stack(single_spiral, 200)
om_star = hnufft.get_rot_stack(single_line, 50, 180)

om_ga_spiral = hnufft.get_golden_angle_rot_stack(single_spiral, 200)
om_ga_star = hnufft.get_golden_angle_rot_stack(single_line, 100)

plt.figure(1)
plt.subplot(221)
plt.scatter(single_line[:, 0], single_line[:, 1])
plt.title('Single line trajectory')

plt.subplot(222)
plt.scatter(single_spiral[:, 0], single_spiral[:, 1])
plt.title('Single spiral trajectory')

plt.subplot(223)
plt.scatter(om_star[::25, 0], om_star[::25, 1], alpha=0.2)
plt.title('Kspace trajectory star')

plt.subplot(224)
plt.scatter(om_spiral[::25, 0], om_spiral[::25, 1], alpha=0.2)
plt.title('Kspace trajectory spiral')
plt.show()

# Now save the reconstruction for different undersampling percentages..
om_name_list = [om_star, om_spiral, om_ga_star, om_ga_spiral]
file_name_list = ['star_sampling', 'spiral_sampling', 'ga_star_sampling', 'ga_spiral_sampling']

for file_name, om in zip(file_name_list, om_name_list):
    # i_sel = 0
    # file_name = file_name_list[i_sel]
    # om = om_name_list[i_sel]
    us_list = []
    us_perc = [95, 90, 80, 60, 40, 20, 0]
    for i_perc in us_perc:
        om_undersampled = hnufft.get_undersampled_traj(om, i_perc/100, Kd[0])
        res_img, res_ksp = hnufft.nufft_to_image(image, om_undersampled, Nd=Nd, Kd=Kd, Jd=Jd)
        us_list.append(res_img)

    title_list = [[str(x) + ' %' for x in us_perc[::-1]]]
    hplotf.plot_3d_list(np.array(us_list)[np.newaxis, ::-1], augm='np.real', subtitle=title_list, ax_off=True)

