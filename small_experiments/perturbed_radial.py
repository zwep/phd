
import data_generator.UndersampledRecon as data_gen
import importlib
import helper.array_transf as harray
import helper.plot_fun as hplotf
import matplotlib.pyplot as plt
# # # # # Testin live undersampled data
import numpy as np
import os

dir_data = '/home/bugger/Documents/data/semireal/rxtx_prostate'
gen = data_gen.DataGeneratorUndersampling(ddata=dir_data, input_shape=(256, 256), dataset_type='test', complex_type='polar',
                                          inhomogeneity='plus',
                                          n_spokes=31, file_ext='npy')

a, b = gen.__getitem__(0)

import helper.plot_fun as hplotf

hplotf.plot_3d_list(a[np.newaxis])
hplotf.plot_3d_list(b)

np.random.standard_normal(10)
np.random.normal(0, 1, 10)


def get_normal(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


import matplotlib.pyplot as plt

Nd = (256, 256)
Kd = (512, 512)
Jd = (6, 6)

x = np.linspace(-1, 1, Kd[0])
y_perturb = get_normal(x, 0, 0.4) * 0.02 * np.sin(25 * x)
y_skewed = 0.1 * np.linspace(0, 1, Kd[0])
plt.plot(y_perturb)

y_line = np.zeros(Kd[0])
x_line = np.linspace(-np.pi, np.pi, Kd[0])
single_line = np.vstack([x_line, y_line]).T
single_line_perb = np.vstack([x_line, y_line + y_perturb + y_skewed]).T
# plt.scatter(single_line[:, 0], single_line[:, 1])
# plt.scatter(single_line_perb[:, 0], single_line_perb[:, 1])

n_spokes = 30
n_lines = 8 * n_spokes  # We have more spokes than we sample. To create some randomness
om_trajectory = hnufft.get_golden_angle_rot_stack(single_line, n_lines)
om_undersampled = hnufft.get_undersampled_traj(om_trajectory, n_undersampled=n_spokes, total_lines=n_lines)

om_trajectory_perb = hnufft.get_golden_angle_rot_stack(single_line_perb, n_lines)
om_undersampled_perb = hnufft.get_undersampled_traj(om_trajectory_perb, n_undersampled=n_spokes, total_lines=n_lines)

plot_om_undersampled = False
if plot_om_undersampled:
    plt_cm = plt.get_cmap('Reds')
    NUM_COLORS = n_spokes
    fig, ax = plt.subplots()
    color_thing = [plt_cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
    for k in range(n_spokes):
        temp_plot = om_undersampled[k * Kd[0]:(k + 1) * Kd[0]]
        ax.scatter(temp_plot[:, 0], temp_plot[:, 1], c=np.array(color_thing[k]).reshape(1, -1))

# And now pull off a resampling...
complex_image = b[0] * np.exp(-1j * b[1])
complex_us, _ = hnufft.nufft_to_image(complex_image, om_undersampled, Nd=Nd, Kd=Kd, Jd=Jd)
complex_us_pertb, _ = hnufft.nufft_to_image(complex_image, om_undersampled_perb, Nd=Nd, Kd=Kd, Jd=Jd)

hplotf.plot_3d_list([complex_us, complex_us_pertb], augm='np.abs')
