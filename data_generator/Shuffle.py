# encoding: utf-8

import torch.utils.data
import numpy as np
import os

import helper.array_transf
import helper.array_transf as harray
import data_generator.Generic as generic_data_generator
import torch.utils.data
import helper_torch.transforms as htransform
import helper.misc as hmisc
import collections
import itertools


"""
Here we have the data generators for the Survey 2 B1 maps

either single channel.. or all channels..
"""

import data_generator.Generic as data_gen


class Shuffle(data_gen.DatasetGenericComplex):
    def __init__(self, ddata, input_shape, target_shape=None, batch_perc=0.010, shuffle=True, dataset_type='train', file_ext='npy', **kwargs):
        input_args = {k: v for k, v in locals().items() if (k !='self') and (k !='__class__')}
        super().__init__(**input_args)

        self.n_slice = kwargs.get('n_slice', 1)

    def __getitem__(self, index):
        """Generate one batch of data"""

        im_t, im_y, im_x = self.img_target_shape

        i_file = self.file_list[index]
        input_file = os.path.join(self.input_dir, i_file)
        temp_input = np.load(input_file)
        n_t, n_y, n_x = temp_input.shape

        # y0 = np.random.randint(0, n_y - im_y + 1)
        # x0 = np.random.randint(0, n_x - im_x + 1)
        y0 = (n_y - im_y) // 2
        x0 = (n_x - im_x) // 2
        n_y_range = np.arange(im_y) + y0
        n_x_range = np.arange(im_x) + x0

        # Subset the matrix to the chosen size..
        temp_input = np.take(temp_input, n_y_range, axis=-2)
        temp_input = np.take(temp_input, n_x_range, axis=-1)
        print(len(n_x_range), len(n_y_range))

        temp_input = temp_input / np.max(temp_input)
        # Perform the shuffeling
        temp_target = shuffle_array(temp_input, n_slice=15)
        # Here is some code to perform shuffeling looped over
        # n_row = input_array.shape[1]
        # # Because we are in three dimensions, we will shuffle each row of the data independently
        # for i in range(n_row):
        #     # Shuffle the group indices
        #     shuffled_group = np.random.permutation(group_index)
        #     if randomness_perc:
        #         shuffled_group = de_randomize_loop(shuffled_group, randomness_perc)
        #     # Remapping
        #     shuffled_index = np.asarray(list(itertools.chain(*[mapping_dict[x] for x in shuffled_group])))
        #     res_shuffle[:, i] = res_shuffle[shuffled_index, i]

        if self.input_is_output:
            temp_target = np.copy(temp_input)

        # return torch.tensor(temp_input).float(), torch.tensor(temp_target).float()
        return temp_input, temp_target


def shuffle_array(input_array, axis=0, n_slice=1, randomness_perc=None, **kwargs):
    """
    Shuffles indices over a given axis. Can set the group size of the shuffeling, and the randomness
    percentage of the shuffle.

    :param input_array:
    :param model_dim:
    :param n_slice:
    :param randomness_perc:
    :param kwargs:
    :return:
    """
    debug = kwargs.get('debug')

    # The axis over which we shuffle will always be the first...
    n_shuffle_axis = input_array.shape[axis]  # axis argument?

    assert n_shuffle_axis % n_slice == 0, "Error with dimension {n_dim}. \n Try these n_slice values {n_opt}".format(
        n_dim=input_array.shape, n_opt=', '.join(map(str, helper.array_transf.simple_div(n_shuffle_axis))))

    if debug:
        # Thus produces a lot of results.. not waiting for it
        print('INFO - SHF: \t Input array shape', input_array.shape)
        print('INFO - SHF: \t n shuffle axis', n_shuffle_axis)
        print('INFO - SHF: \t n slice', n_slice)

    # Group mapping
    # This determines the difficulty of the problem. More groups, is more complex
    group_index = np.arange(n_slice)
    # This is how often one group occurs.
    multiplicity_group = n_shuffle_axis // n_slice
    # Now we translate this into a mapping array. Where we see the relation between the groups and positions
    mapping_array = np.array(list(zip(np.repeat(group_index, multiplicity_group), np.arange(n_shuffle_axis))))  ## !!

    # Transform it to a dictionary
    mapping_dict = collections.defaultdict(list)
    [mapping_dict[key].append(val) for key, val in mapping_array]

    # Shuffle the group indices
    shuffled_group = np.random.permutation(group_index)
    if randomness_perc:
        shuffled_group = de_randomize_loop(shuffled_group, randomness_perc)
    # Remapping
    shuffled_index = np.asarray(list(itertools.chain(*[mapping_dict[x] for x in shuffled_group])))
    res_shuffle = np.take(input_array, shuffled_index, axis=axis)

    return res_shuffle, shuffled_index


def de_randomize_loop(x_random, pc_goal, n_iter_max=200, eps=0.05):
    """
    This might need some explanation.. We try to measure and control HOW random something is.

    The measure for this is the distance from its original position.. and this is manipulated and normalized in such
    a way that we can probably use it over difference sequence length.
    It is normalized by the maximum amount of randomness we can achieve by this metric. This comes as a simple
    formula expressed as x_max_random.

    :param x_random:
    :param pc_goal:
    :param n_iter_max:
    :param eps:
    :return:
    """
    N = len(x_random)
    x_orig = np.arange(N)
    x_max_random = (N * (N - 1) / 2) / ((N - 1) * 2)
    x_temp, dest_int, orig_int = de_randomize(x_random)
    metric_random_init = (np.sum(np.abs(x_orig - np.argsort(x_temp))) / 2 - 1) / N + 1 / N
    metric_random = metric_random_init
    n_iter = 0
    c_goal = pc_goal * x_max_random

    while n_iter < n_iter_max and np.abs(metric_random - c_goal) >= eps:
        n_iter += 1
        if metric_random - c_goal < 0:
            # If we failed to get closer to the goal.. (However... is this properly done..?)
            # un-do this point, and try another
            # It can be shown that this method is far superior than keep trying random guesses.
            x_temp[[dest_int, orig_int]] = x_temp[[orig_int, dest_int]]
            x_temp, dest_int, orig_int = de_randomize(x_temp)
        else:
            # Switch two positions..
            x_temp, dest_int, orig_int = de_randomize(x_temp)

        # Check if the switching created a more or less random array...
        metric_random = (np.sum(np.abs(x_orig - np.argsort(x_temp))) / 2 - 1) / N + 1 / N

    return x_temp


def de_randomize(x):
    x = np.array(list(x))
    dest_int = np.random.randint(len(x))
    found_int = np.where(x == dest_int)
    if len(found_int[0]):
        # print(found_int, len(found_int[0]))
        orig_int = found_int[0][0]
    else:
        orig_int = dest_int
    # Here we implicitly say where the integer needs to be.
    x[[orig_int, dest_int]] = x[[dest_int, orig_int]]

    return x, dest_int, orig_int


if __name__ == "__main__":
    ddata = '/home/bugger/Documents/data/leiner_numpy_test'
    input_shape = (30, 128, 128)

    import importlib
    import data_generator.Shuffle as data_gen
    importlib.reload(data_gen)
    A = data_gen.Shuffle(ddata=ddata, input_shape=input_shape, n_slice=1)
    a, b = A.__getitem__(0)

    A = data_gen.Shuffle(ddata=ddata, input_shape=input_shape, n_slice=30)
    a1, b1 = A.__getitem__(0)

    import helper.plot_fun as hplotf
    import helper.plot_class as hplotc
    hplotc.SlidingPlot(a)
    hplotc.SlidingPlot(a1)
    hplotc.SlidingPlot(a1-a)