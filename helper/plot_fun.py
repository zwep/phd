# encoding: utf-8

"""
Here we have a collection of plotting functionalities
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import helper.misc as hmisc
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import helper.array_transf as harray
import os




def plot_feature_star(x_data, x_label, fc='r', fig=None, ax_index=0, max_value=None):
    # Used for plotting
    xy_data_int = np.ceil(x_data).astype(int)
    if max_value is None:
        max_value = np.max(xy_data_int)

    n_points = len(x_data)
    theta_angles = np.linspace(0, 2 * np.pi - (2 * np.pi/n_points), n_points) + np.pi/3
    xy_data = [np.array([x, 0]) for x in x_data]
    xy_rot_data = np.array([harray.rot2d(i_angle) @ x_vec for i_angle, x_vec in zip(theta_angles, xy_data)])
    line_data = np.array([harray.rot2d(i_angle) @ np.array([max_value, 0]) for i_angle in theta_angles])

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[ax_index]

    ax.scatter(xy_rot_data[:, 0], xy_rot_data[:, 1], c='k')
    for i_coord in xy_rot_data:
        line_obj = plt.Line2D([0, i_coord[0]], [0, i_coord[1]], color='black', alpha=0.5, linestyle='--')
        ax.add_line(line_obj)
    for ii, i_coord in enumerate(line_data):
        line_obj = plt.Line2D([0, i_coord[0]], [0, i_coord[1]], color='black', alpha=0.25, linestyle='--')
        ax.text(*i_coord, x_label[ii])
        ax.add_line(line_obj)

    poly_obj = plt.Polygon(xy_rot_data, alpha=0.5, fc=fc)

    for iradius in range(1, max_value+1):
        circle_obj = plt.Circle(xy=(0, 0), radius=iradius, fc='b', zorder=0, linestyle='--', alpha=0.1)
        if iradius == max_value:
            ax.text(0, 0.95 * iradius, f'{100 * iradius}%', zorder=3)
        else:
            ax.text(0, iradius, f'{100 * iradius}%', zorder=3)
        ax.add_patch(circle_obj)

    ax.add_patch(poly_obj)
    ax.set_xlim(-max_value-0.05, max_value+0.05)
    ax.set_ylim(-max_value-0.05, max_value+0.05)

    return fig


def add_text_box_left(figure_obj, ax_index, string_text):
    ax_obj = figure_obj.axes[ax_index]
    x0, y0, width, height = ax_obj.get_position().bounds
    width_rect = 0.025
    rectangle_obj = matplotlib.patches.Rectangle((x0 - width_rect, y0), width_rect, height, edgecolor='black',
                                                 linewidth=3, fill=True, facecolor='white', alpha=1.0,
                                                 transform=figure_obj.transFigure, figure=figure_obj)
    figure_obj.patches.extend([rectangle_obj])
    figure_obj.text(x0 - width_rect / 2, y0 + height / 2, string_text, fontsize=14, rotation='vertical',
                    verticalalignment='center',
                    horizontalalignment='center', color='black')


def add_text_box(figure_obj, ax_index, string_text, height_rect=None, height_offset=0, **kwargs):
    position = kwargs.get('position', 'top')
    fontsize = kwargs.get('fontsize', 14)
    linewidth = kwargs.get('linewidth', 3)
    # position: top or bottom
    ax_obj = figure_obj.axes[ax_index]
    x0, y0, width, height = ax_obj.get_position().bounds
    if height_rect is None:
        height_rect = 0.05

    if position == 'top':
        rectangle_y_coord = y0 + height + height_offset  # top
        text_y_coord = y0 + height + height_rect/2 + height_offset  # top
    elif position == 'bottom':
        rectangle_y_coord = y0 - height_rect + height_offset  # bottom
        text_y_coord = y0 - height_rect/2 + height_offset  # bottom
    else:
        rectangle_y_coord = y0
        text_y_coord = y0
        print(f'Unknown position. Received: {position}')
        print('Please use `top` or `bottom`.')
    rectangle_obj = matplotlib.patches.Rectangle((x0, rectangle_y_coord), width, height_rect, edgecolor='black',
                                                 linewidth=linewidth, fill=True, facecolor='white', alpha=1.0,
                                                 transform=figure_obj.transFigure, figure=figure_obj)
    figure_obj.patches.extend([rectangle_obj])
    figure_obj.text(x0 + width / 2, text_y_coord, string_text, fontsize=fontsize, rotation='horizontal',
                    verticalalignment='center',
                    horizontalalignment='center', color='black')


def plot_multi_lines(data, cmap_col='Reds', style='.', x_range=None, legend=False):
    """
    Example...

    import numpy as np
    import matplotlib.pyplot as plt

    N = 100
    n_line = 10
    x = np.linspace(0, 2 * np.pi, N)
    noise = np.random.rand(N)
    y = np.sin(x)
    y_set = [y + noise ** np.random.rand() for i in range(n_line)]
    y_set = np.vstack(y_set).T
    y_set.shape

    plot_multi_lines(y_set)
    plt.show()
    :param data:
    :param cmap_col:
    :return:
    """
    num_lines = data.shape[1]
    plt_cm = plt.get_cmap(cmap_col)
    color_list = [plt_cm(1. * i / (num_lines+1)) for i in range(1, num_lines+1)]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', color_list)
    for i in range(num_lines):
        if x_range is not None:
            ax.plot(x_range, data[:, i], marker=style, label=i)
        else:
            ax.plot(data[:, i], marker=style, label=i)

    if legend:
        ax.legend()
    return fig


def plot_multi_points(data, cmap_col='Reds'):
    """

    :param data: nd-array of size n_data x 2
    :param cmap_col: 'Reds'
    :param legend: binary, yes or no legend
    :param alpha: the.. transparancy
    :return:
    """

    num_lines = data.shape[0]
    plt_cm = plt.get_cmap(cmap_col)
    color_list = [plt_cm(1. * i / num_lines) for i in range(num_lines)]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', color_list)
    for i in range(num_lines):
        ax.scatter(data[i, 0], data[i, 1], label=i)


def plot_surf(data):
    # Expect input: (x,y)
    from mpl_toolkits.mplot3d import Axes3D

    # Set up grid and test data
    nx, ny = data.shape
    x = range(nx)
    y = range(ny)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X.T, Y.T, data)
    # We could use plot_surface here.. or contourf with offset to do something fancy


def plot_3d_list(image_list, **kwargs):
    # Input of either a 2d list of np.arrays.. or a 3d list of np.arrays..
    figsize = kwargs.get('figsize')
    fignum = kwargs.get('fignum')
    dpi = kwargs.get('dpi')

    title_string = kwargs.get('title', "")
    sub_title = kwargs.get('subtitle', None)
    cbar_ind = kwargs.get('cbar', False)
    cmap = kwargs.get('cmap', 'gray')

    vmin = kwargs.get('vmin', None)
    ax_off = kwargs.get('ax_off', False)
    augm_ind = kwargs.get('augm', None)
    aspect_mode = kwargs.get('aspect', 'equal')

    wspace = kwargs.get('wspace', 0.1)
    hspace = kwargs.get('hspace', 0.1)
    debug = kwargs.get('debug', False)

    f = plt.figure(fignum, figsize, dpi)
    f.suptitle(title_string)

    # Only when we have an numpy array
    if isinstance(image_list, np.ndarray):
        # With just two dimensions..
        if image_list.ndim == 1:
            image_list = image_list[np.newaxis, np.newaxis]
        elif image_list.ndim == 2:
            # Add one..
            image_list = image_list[np.newaxis]


    n_rows = len(image_list)
    gs0 = gridspec.GridSpec(n_rows, 1, figure=f)
    gs0.update(wspace=wspace, hspace=hspace)  # set the spacing between axes.

    for i, i_gs in enumerate(gs0):
        temp_img = image_list[i]

        if hasattr(temp_img, 'ndim') and hasattr(temp_img, 'shape') and hasattr(temp_img, 'reshape'):
            if temp_img.ndim == 4:
                n_sub_col = temp_img.shape[0]
                n_sub_row = temp_img.shape[1]
                temp_img = temp_img.reshape((n_sub_col * n_sub_row, ) + temp_img.shape[2:])
            elif temp_img.ndim == 3:
                n_sub_col = temp_img.shape[0]
                if n_sub_col > 8:
                    n_sub_col, n_sub_row = hmisc.get_square(n_sub_col)
                else:
                    n_sub_row = 1
            elif temp_img.ndim == 2:
                temp_img = temp_img[np.newaxis]
                n_sub_col = 1
                n_sub_row = 1
            else:
                temp_img = temp_img[np.newaxis, np.newaxis]
                n_sub_col = 1
                n_sub_row = 1
        else:
            n_sub_col = len(temp_img)
            n_sub_row = 1

        # If we want to specifcy the vmin per list item.. we can do that here..
        if isinstance(vmin, list):
            sel_vmin = vmin[i]
        else:
            sel_vmin = vmin

        for j, ii_gs in enumerate(i_gs.subgridspec(n_sub_row, n_sub_col)):

            ax = f.add_subplot(ii_gs)
            if augm_ind:
                plot_img = eval('{fun}({var})'.format(fun=augm_ind, var=str('temp_img[j]')))
                if 'angle' in augm_ind:
                    sel_vmin = (-np.pi, np.pi)
            else:
                plot_img = temp_img[j]

            if debug:
                print(f'shape {i} - {temp_img.shape}', end=' \t|\t')
                print(f'row/col {n_sub_row} - {n_sub_col}', end=' \t|\t')
                print(f'shape {j} - {plot_img.shape}', end=' \t|\n')

            if plot_img.shape[0] == 1:
                map_ax = ax.plot(plot_img.ravel())
            else:
                map_ax = ax.imshow(plot_img, vmin=sel_vmin, aspect=aspect_mode, cmap=cmap)

            if cbar_ind:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                temp_cbar = plt.colorbar(map_ax, cax=cax)
                if sel_vmin is None:
                    vmin_temp = [plot_img.min(), plot_img.max()]
                    vmin_temp = list(map(float, vmin_temp))
                    map_ax.set_clim(vmin_temp)
                    temp_cbar.set_ticks(vmin_temp)
                else:
                    map_ax.set_clim(sel_vmin)
                    temp_cbar.set_ticks(sel_vmin)

            if sub_title is not None:
                ax.set_title(sub_title[i][j])
            if ax_off:
                ax.set_axis_off()

    return f


def get_all_mid_slices(image_array, offset=(0, 0, 0)):
    image_shape = np.array(image_array.shape)//2
    image_ndim = image_array.ndim
    ax_combinations = list(itertools.combinations(range(image_ndim), 2))
    sliced_array = []
    for i_ax_comb in ax_combinations:
        slice_list = [slice(None)] * image_ndim
        remaining_ax = set(range(image_ndim)).difference(set(i_ax_comb))
        # Create a list of slices..
        for i_ax in list(remaining_ax):
            slice_list[i_ax] = image_shape[i_ax] + offset[i_ax]
        sliced_array.append(image_array[tuple(slice_list)])
    return sliced_array

"""
def change_tick_ax(ax, nx, ny):
    # source https://stackoverflow.com/questions/38973868/adjusting-gridlines-and-ticks-in-matplotlib-imshow
    # Major ticks
    ax.set_xticks(np.arange(0, nx, 1))
    ax.set_yticks(np.arange(0, ny, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, nx+1, 1))
    ax.set_yticklabels(np.arange(1, ny+1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, nx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, ny, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax.axis('off')
    return ax
    
"""


# Added this one so that I dont need to keep importing matplotlib..
def close_all():
    plt.close('all')

