import numpy as np
import helper.array_transf as harray
import sigpy.mri
import matplotlib.pyplot as plt
import scipy.io
import helper.misc as hmisc
import helper.plot_class as hplotc
import os

"""
This should contain generic helpers for reconstruction

"""


def get_spokes_from_sin_file(sin_file):
    # Get the number of spokes from the sin file
    with open(sin_file, 'r') as file_obj:
        sin_list = file_obj.readlines()
    retro_max_intp_line = [x for x in sin_list if 'retro_max_intp_length' in x][0]
    print('Single line ', retro_max_intp_line)
    isolated_number = retro_max_intp_line.strip().split(':')[-1].strip()
    print('Isolated number ', isolated_number, type(isolated_number))
    n_spokes = int(isolated_number)
    return n_spokes


def get_key_from_sin_file(sin_file, key):
    # Get generic key from sin file
    with open(sin_file, 'r') as file_obj:
        sin_list = file_obj.readlines()
    found_key = [x for x in sin_list if key in x][0]
    found_key = found_key.strip().split(':')[-1].strip()
    return found_key


def get_golden_angle_spokes(n_spokes):
    increment = 180 * (3 - np.sqrt(5))
    acq_angles = (np.arange(n_spokes) * increment) % 360
    return acq_angles


def get_angle_spokes(n_spokes):
    increment = 360 / n_spokes  # in degree
    mod = 0
    first_angle = 0
    second_angle = 180 - increment / 2
    # If we have odd spokes.. we need one more with the even spokes...
    odd_spokes = np.deg2rad(np.arange(second_angle, second_angle - n_spokes // 2 * increment, -increment))

    if n_spokes % 2:
        mod = 1
        odd_spokes = np.append(odd_spokes, [0])
    even_spokes = -np.deg2rad(np.arange(first_angle, n_spokes // 2 * increment + mod, increment))
    acq_angles = hmisc.interleave_two_list(even_spokes, odd_spokes)

    return acq_angles[:n_spokes]


def get_trajectory_sin_file(sin_file, golden_angle=False):
    min_encoding = get_key_from_sin_file(sin_file, 'non_cart_min_encoding_nrs').split()
    max_encoding = get_key_from_sin_file(sin_file, 'non_cart_max_encoding_nrs').split()
    trajectory = get_trajectory_n_spoke(min_encoding=min_encoding, max_encoding=max_encoding, golden_angle=golden_angle)
    return trajectory


def get_n_spoke(sin_file):
    min_encoding = get_key_from_sin_file(sin_file, 'non_cart_min_encoding_nrs').split()
    max_encoding = get_key_from_sin_file(sin_file, 'non_cart_max_encoding_nrs').split()
    n_spokes = int(max_encoding[1]) - int(min_encoding[1]) + 1
    return n_spokes


def get_trajectory_n_spoke(min_encoding, max_encoding, golden_angle=False):
    # min/max encoding is a 4-D vector. Dimensions correspond to x, y, z, ..?
    # Result here is (nspoke, nsample, ndim)
    n_spokes = int(max_encoding[1]) - int(min_encoding[1]) #+ 1
    if golden_angle:
        acquisition_angles = get_golden_angle_spokes(n_spokes)
    else:
        acquisition_angles = get_angle_spokes(n_spokes)

    base_trajectory_x = np.zeros(int(max_encoding[0]) - int(min_encoding[0])) #  + 1
    base_trajectory_y = np.arange(int(min_encoding[0]), int(max_encoding[0])) # + 1
    base_trajectory = np.stack([base_trajectory_x, base_trajectory_y], axis=1)

    trajectory = []
    for i in range(n_spokes):
        rot_max = harray.rot2d(-acquisition_angles[i])
        temp_traj = np.matmul(base_trajectory, rot_max)
        trajectory.append(temp_traj)

    return np.array(trajectory)


def get_trajectory(img_size):
    N = max(img_size)
    max_spokes = int(np.ceil((np.pi / 2) * N))
    trajectory_radial = sigpy.mri.radial(coord_shape=(max_spokes, N, 2), img_shape=img_size, golden=False)
    return trajectory_radial


def reconstruct_unsorted_radial_data(unsorted_array, sin_file):
    # We need an unsorted array (nx, nchron)
    # Where nchron are all the data acquisitions in a chronological order
    # The sin file is associated with the give (numpy) array and tells us more about the recon parameters
    trajectory = get_trajectory_sin_file(sin_file)
    n_spokes = trajectory.shape[0]
    ovs = float(get_key_from_sin_file(sin_file, 'non_cart_grid_overs_factor'))
    width = int(get_key_from_sin_file(sin_file, 'non_cart_grid_kernel_size'))
    n_coil = int(get_key_from_sin_file(sin_file, 'nr_channel_names'))
    n_card = int(get_key_from_sin_file(sin_file, 'nr_cardiac_phases'))
    dcf = np.sqrt(trajectory[:, :, 0] ** 2 + trajectory[:, :, 1] ** 2)
    result = []
    for i_coil in range(n_coil):
        # This gets.... one phase...?
        selected_data = unsorted_array[:, i_coil::n_coil][:, :n_spokes]
        selected_data = np.moveaxis(selected_data, -1, 0)
        temp_img = sigpy.nufft_adjoint(selected_data, coord=trajectory, oversamp=ovs, width=width)
        result.append(temp_img)

    return np.array(result)