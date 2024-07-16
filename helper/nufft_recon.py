# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf


def get_rot_stack(x, n, max_degree=360, random_offset=None):
    # x = x_range
    # input of shape derp x 2
    # random_offset.. if not None: a max value. Random vaules are taken from 0..1
    rot_x = []
    for i_degree in np.linspace(0, max_degree, n):
        theta = np.radians(i_degree)
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array(((c, -s), (s, c)))
        temp = np.dot(x, rot_mat)
        if random_offset is not None:
            temp_random = (2 * np.random.rand(1, 2) - 1) * random_offset
            temp += temp_random
        rot_x.append(temp)
    return np.vstack(rot_x)


def get_golden_angle_rot_stack(x, n, golden_angle=180/1.618):
    # x.. this is the single line... or curve
    # n.. this is the amount of curves we have
    rot_x = []
    for i_degree in np.linspace(0, golden_angle*(n-1), n):
        theta = np.radians(i_degree)
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array(((c, -s), (s, c)))
        temp = np.dot(x, rot_mat)
        rot_x.append(temp)
    return np.vstack(rot_x)


def get_approx_golden_angle_rot_stack(x, n, golden_angle=180/1.618, error=3):
    # x.. this is the single line... or curve
    # n.. this is the amount of curves we have
    # error.. in number of degrees
    rot_x = []
    for i_degree in np.linspace(0, golden_angle*(n-1), n):
        theta = np.radians(i_degree + np.random.uniform(-error, error))
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array(((c, -s), (s, c)))
        temp = np.dot(x, rot_mat)
        rot_x.append(temp)
    return np.vstack(rot_x)


def get_approx_rigid_golden_angle_rot_stack(x, n, golden_angle=180/1.618, error=3, rigid_motion=None):
    # x.. this is the single line... or curve
    # n.. this is the amount of curves we have
    # error.. in number of degrees
    rot_x = []
    for i_degree in np.linspace(0, golden_angle*(n-1), n):
        theta = np.radians(i_degree + np.random.uniform(-error, error))
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array(((c, -s), (s, c)))
        if rigid_motion is None:
            temp = np.dot(x, rot_mat)
        else:
            temp = np.dot(x + rigid_motion, rot_mat)
        rot_x.append(temp)

    return np.vstack(rot_x)


def get_undersampled_traj(x_traj, n_undersampled, total_lines):
    x_traj_split = np.array(np.split(x_traj, total_lines))
    random_lines = np.random.choice(range(total_lines), size=n_undersampled, replace=False)
    x_traj_undersampled = x_traj_split[random_lines].reshape(-1, 2)

    return x_traj_undersampled, random_lines


if __name__ == "__main__":
    # Lets try some images...
    import skimage.data as data
    import matplotlib.pyplot as plt

    A = data.astronaut()[:, :, 0]
    nx, ny = A.shape
    x_range = np.arange(nx//2)
    y_range = np.zeros(nx//2)
    x_line = np.stack([x_range, y_range], axis=1)
