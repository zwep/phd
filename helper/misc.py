# encoding: utf-8

"""
Some misc helpers
"""
import yaml
import itertools
import shutil
import sys
import warnings
import math
import numpy as np
import os
import copy
import struct
import scipy.stats
from skimage.metrics import structural_similarity
import skimage.transform as sktransf
import helper.array_transf as harray
import imageio
from skimage.util import img_as_ubyte

from helper.array_transf import simple_div
import pathlib
import h5py
import nibabel
import pydicom
import collections
from PIL import Image
import scipy.io
import json
from typing import List, Tuple

def get_single_file(dd, ext, file_index=0):
    file_list = os.listdir(dd)
    file_list = [x for x in file_list if x.endswith(ext)]
    return os.path.join(dd, file_list[file_index])


def lower_bound_line(points: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[int]]:
    # Sort the points by x coordinate
    points.sort(key=lambda p: p[0])
    # Initialize a list to store the points on the lower bound line
    lower_bound = []
    index_lower_bound = []
    # Add the first point to the lower bound line
    lower_bound.append(points[0])
    # Iterate through the remaining points
    for i in range(1, len(points)):
        # Get the current point and the previous point on the lower bound line
        p, prev = points[i], lower_bound[-1]
        # If the y coordinate of the current point is less than the y coordinate of the previous point on the lower bound line,
        # add the current point to the lower bound line
        if p[1] < prev[1]:
            lower_bound.append(p)
            # Also store the index, so that we can back-track where the results came from
            index_lower_bound.append(i)
    # Return the lower bound line
    return lower_bound, index_lower_bound


def get_minimum_curve(x_coords, y_coords, n_points=20):
    if isinstance(x_coords, list):
        x_coords = np.array(x_coords)
    if isinstance(y_coords, list):
        y_coords = np.array(y_coords)
    # Not sure yet how to resolve the interval problem...
    min_x = x_coords.min()
    max_x = x_coords.max()
    x_interval = np.linspace(min_x, max_x * 1.01, n_points)
    new_x_coords = []
    new_y_coords = []
    new_min_index_coords = []
    for i in range(n_points - 1):
        inbetween_index = (x_interval[i] <= x_coords) * (x_coords < x_interval[i + 1])
        # Only check when we find something..
        if sum(inbetween_index):
            # Find which indices we need (global)
            global_min_index_array = np.argwhere(inbetween_index).ravel()
            # Get the index of the minimum y coordinate
            local_min_index = np.argmin(y_coords[inbetween_index])
            # Translate this (local) index to the global array
            global_min_index = global_min_index_array[local_min_index]
            # Retrieve the new x and y coords
            temp_y_coords = y_coords[global_min_index]
            temp_x_coords = x_coords[global_min_index]
            new_min_index_coords.append(global_min_index)
            # Plot stuff...
            # plt.scatter(x_coords[inbetween_index], y_coords[inbetween_index])
            # plt.hlines(y=temp_y_coords, xmin=new_x_coords[i], xmax=new_x_coords[i + 1])
        else:
            # Just append the last known value
            temp_y_coords = new_y_coords[-1]
            temp_x_coords = new_x_coords[-1]
            print(f'No values here.. {x_interval[i].round(2)} < x < {x_interval[i + 1].round(2)}')
        new_y_coords.append(temp_y_coords)
        new_x_coords.append(temp_x_coords)

    return new_x_coords, new_y_coords, new_min_index_coords


def get_maximum_curve(x_coords, y_coords, n_points=20):
    """
    Find the maximum value of y for each interval of x values, and return the
    corresponding x and y coordinates as well as the indices of the original
    coordinates array.
    """
    if isinstance(x_coords, list):
        x_coords = np.array(x_coords)
    if isinstance(y_coords, list):
        y_coords = np.array(y_coords)

    # create an array of x values spaced evenly between the minimum and maximum x value with n_points
    min_x = x_coords.min()
    max_x = x_coords.max()
    new_x_coords = np.linspace(min_x, max_x * 1.01, n_points)
    new_y_coords = []
    new_max_index_coords = []
    for i in range(n_points - 1):
        # find the indices where the x value is within the current interval
        inbetween_index = (new_x_coords[i] <= x_coords) * (x_coords < new_x_coords[i + 1])
        if sum(inbetween_index):
            # find the global indices of the x values within the current interval
            global_max_index_array = np.argwhere(inbetween_index).ravel()
            # find the index of the maximum y value within the current interval
            local_max_index = np.argmax(y_coords[inbetween_index])
            # translate the local index to the global index
            global_max_index = global_max_index_array[local_max_index]
            # retrieve the corresponding y value
            temp_y_coords = y_coords[inbetween_index][local_max_index]
            new_max_index_coords.append(global_max_index)
            # Plot stuff...
            # plt.scatter(x_coords[inbetween_index], y_coords[inbetween_index])
            # plt.hlines(y=temp_y_coords, xmin=new_x_coords[i], xmax=new_x_coords[i + 1])
        else:
            # Just append the last known value
            temp_y_coords = new_y_coords[-1]
            temp_x_coords = new_x_coords[-1]
            print(f'No values here.. {new_x_coords[i]} < x < {new_x_coords[i + 1]}')
        new_y_coords.append(temp_y_coords)
    return new_x_coords, new_y_coords, new_max_index_coords


def get_mean_curve(x_coords, y_coords, n_points=20):
    if isinstance(x_coords, list):
        x_coords = np.array(x_coords)
    if isinstance(y_coords, list):
        y_coords = np.array(y_coords)
    # Not sure yet how to resolve the interval problem...
    min_x = x_coords.min()
    max_x = x_coords.max()
    new_x_coords = np.linspace(min_x, max_x * 1.01, n_points)
    new_y_coords = []
    for i in range(n_points - 1):
        inbetween_index = (new_x_coords[i] <= x_coords) * (x_coords < new_x_coords[i + 1])
        if sum(inbetween_index):
            global_min_index_array = np.argwhere(inbetween_index).ravel()
            # Take the mean value of the y index
            temp_y_coords = np.mean(y_coords[inbetween_index])
            # Plot stuff...
            # plt.scatter(x_coords[inbetween_index], y_coords[inbetween_index])
            # plt.hlines(y=temp_y_coords, xmin=new_x_coords[i], xmax=new_x_coords[i + 1])
        else:
            # Just append the last known value
            temp_y_coords = new_y_coords[-1]
            temp_x_coords = new_x_coords[-1]
            print(f'No values here.. {new_x_coords[i]} < x < {new_x_coords[i + 1]}')
        new_y_coords.append(temp_y_coords)
    return new_x_coords, new_y_coords


def print_dtype_names(x, sep=0):
    # This can be useful for exploring .mat files...
    name_list = x.dtype.names
    if name_list is not None:
        for i_name in name_list:
            print(sep*'\t', i_name)
            # This selection of [0][0] is very typical to .mat objects..
            print_dtype_names(x[i_name][0][0], sep+1)


def store_json(x_dict, ddir):
    serialized_dict = json.dumps(x_dict)
    with open(ddir, 'w') as f:
        f.write(serialized_dict)

# Create an alias
write_json = store_json


def read_json(ddata):
    with open(ddata, 'r') as f:
        temp = f.read()

    loaded_json = json.loads(temp)
    return loaded_json


def load_dir(x, filter_string='', n_limit=np.inf, take_axis=-1, move_axis=False):
    all_array = []
    counter = 0
    for i_file in os.listdir(x):
        if filter_string in i_file:
            if counter > n_limit:
                break
            file_path = os.path.join(x, i_file)
            loaded_array = load_array(file_path)
            print('Shape of array ', loaded_array.shape)
            if take_axis is not None:
                sel_slice = loaded_array.shape[take_axis]//2
                sel_array = np.squeeze(np.take(loaded_array, sel_slice, axis=take_axis))
            else:
                sel_array = loaded_array
            all_array.append(sel_array)
            counter += 1
    return all_array


def interleave_two_list(x, y):
    return [val for pair in zip(x, y) for val in pair]


def load_json(file_path):
    with open(file_path, 'r') as f:
        temp = f.read()
    temp_json = json.loads(temp)
    return temp_json


def load_yaml(file_path):
    with open(file_path, "r") as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_file


def write_yaml(yaml_dict, file_path):
    # Write YAML file
    with open(file_path, 'w') as outfile:
        yaml.dump(yaml_dict, outfile, default_flow_style=False, allow_unicode=True)


def get_base_name(file_name):
    base_name = pathlib.Path(file_name)
    for _ in base_name.suffixes:
        base_name = base_name.with_suffix('')

    return base_name.name


def get_ext(file_name):
    base_name = pathlib.Path(file_name)
    ext = ''.join(base_name.suffixes)
    return ext


def convert_image_to_gif(input_image, output_path, n_slices=None, nx=128, ny=128, duration=None):
    if n_slices is None:
        n_slices = input_image.shape[0]
    input_image = sktransf.resize(input_image, output_shape=(n_slices, nx, ny), preserve_range=True, anti_aliasing=False)
    input_image = harray.scale_minmax(input_image)
    input_image = img_as_ubyte(input_image)
    input_image = np.abs(input_image)
    imageio.mimsave(output_path, input_image, duration=duration)


def patch_min_fun(scale, x, y):
    # Different optimizer for the homogeneize
    hist_x, _ = np.histogram(scale * x, range=(1e-8, 1 - 1e-8), bins=256, density=True)
    hist_y, _ = np.histogram(y, bins=256, range=(1e-8, 1 - 1e-8), density=True)

    wasserstein_dist = scipy.stats.wasserstein_distance(hist_x, hist_y)

    return wasserstein_dist


def patch_min_fun_ssim(scale, x, y):
    # Optimizer for the patch homogenizer
    ssim_dist = structural_similarity(scale * x, y, data_range=1)
    return -ssim_dist


def patch_min_fun_l2(scale, x, y):
    # Optimizer for the patch homogenizer
    l2_dist = np.linalg.norm(scale * x - y)
    return l2_dist


def get_file_containing_str(directory, file_str='', ext=None, n=1):
    # Get the latest modified file with 'file_filter' in its name
    if ext is not None:
        file_list = [os.path.join(directory, x) for x in os.listdir(directory) if file_str in x and x.endswith(ext)]
    else:
        file_list = [os.path.join(directory, x) for x in os.listdir(directory) if file_str in x]
    return file_list[:n]


def find_file_in_dir(str_name, dir_name, ext='sin'):
    for d, _, f in os.walk(dir_name):
        filter_f = [x for x in f if x.endswith(ext)]
        if len(filter_f) > 0:
            for i_file in filter_f:
                if str_name in i_file:
                    return os.path.join(d, i_file)
                else:
                    continue

def find_all_files_in_dir(str_name, dir_name, ext='sin'):
    found_files = []
    for d, _, f in os.walk(dir_name):
        filter_f = [x for x in f if x.endswith(ext)]
        if len(filter_f) > 0:
            for i_file in filter_f:
                if str_name in i_file:
                    temp = os.path.join(d, i_file)
                    found_files.append(temp)
                else:
                    continue
    return found_files

def get_latest_file(directory, file_filter='', fun=os.path.getmtime, n=1):
    # Get the latest modified file with 'file_filter' in its name
    file_list = [os.path.join(directory, x) for x in os.listdir(directory) if file_filter in x]
    latest_file = sorted(file_list, key=fun)[-n:]
    return latest_file


def get_magic_number(file):
    with open(file, 'rb') as f:
        print(struct.unpack('<H', f.read(2)))


def flatten(L):
    # Got it from: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    # Yields pretty nice results
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def convert_remote2local_dict(model_path, path_prefix, name='config_param.json'):
    # Used to read and change the data path to something that we have locally...
    with open(os.path.join(model_path, name), 'r') as f:
        temp_text = f.read()
        config_model = json.loads(temp_text)

    config_model['dir']['doutput'] = model_path
    config_model['dir']['dtemplate'] = model_path
    temp_data = os.path.join(path_prefix, os.path.basename(config_model['dir']['ddata']))
    config_model['dir']['ddata'] = temp_data
    return config_model


def get_size_of_locals(nspace=40):
    # Used to get the byte size of local variables to debug memory leaks..
    derp = globals()
    imported_lib = sys.modules.keys()
    for x, y in derp.items():
        if x in imported_lib:
            print(x)
        else:
            print(x, end=(nspace - len(x)) * ' ')
            print(sys.getsizeof(y))


def correct_batch_size(batch_size, n_files):
    if n_files % batch_size:
        # If the batch size is not a divisor of the amount of files.. change this.
        # Look for a candidate in the region of divisors of the amount of files..
        div_nfile = simple_div(n_files)
        ind_min = np.argmin(np.abs([batch_size - x for x in div_nfile]))
        batch_size = div_nfile[ind_min]
    return batch_size


def compare_dict(a, b):
    # Compared two dictionaries..
    # Posts things that are not equal..
    res_compare = []
    for k in set(list(a.keys()) + list(b.keys())):
        if isinstance(a[k], dict):
            z0 = compare_dict(a[k], b[k])
        else:
            z0 = a[k] == b[k]

        z0_bool = np.all(z0)
        res_compare.append(z0_bool)
        if not z0_bool:
            print(k, a[k], b[k])
    return np.all(res_compare)


def create_datagen_dir(dest_dir, type_list=('test', 'validation', 'train'), data_list=('input', 'target', 'mask')):
    # Used to create [default]
    # dest_dir
    #   test
    #       input
    #       target
    #       mask
    #   train
    #       input
    #       target
    #       mask
    #   validation
    #       input
    #       target
    #       mask

    information_container = {}
    if not os.path.isdir(dest_dir):
        print('Creating dir ', dest_dir)
        os.makedirs(dest_dir)

    for i_type in type_list:
        temp_path = os.path.join(dest_dir, i_type)
        if not os.path.isdir(temp_path):
            print('\t Creating dir ', temp_path)
            os.mkdir(temp_path)
        information_container[i_type] = {}
        for i_data in data_list:
            temp_path_0 = os.path.join(temp_path, i_data)
            # Strip the last / in case data_list is only an empty string
            if temp_path_0.endswith('/'):
                temp_path_0 = temp_path_0[:-1]
            if not os.path.isdir(temp_path_0):
                print('\t\t Creating dir ', temp_path_0)
                os.mkdir(temp_path_0)

            information_container[i_type][i_data] = temp_path_0

    return information_container


def create_train_test_val_files(input_files, train_perc=0.70, validation_perc=0.10, test_perc=0.20):
    n_files = len(input_files)
    n_train = int(n_files * train_perc)
    n_validate = int(n_files * validation_perc)
    n_test = int(n_files * test_perc)
    list_train = input_files[0:n_train]
    list_validation = input_files[n_train: (n_train + n_validate)]
    list_test = input_files[-n_test:]
    return {'train': list_train, 'test': list_test, 'val': list_validation, 'validation': list_validation}


def create_and_copy_data_split(source_dir, target_dir, sel_target_type='input', data_list=None):
    if data_list is None:
        data_list = ('input', 'target')
    # Source_dir is the location that contains all the files
    # These will be split according to 70%, 10%, 20%
    # and put under the target_dir train/test/val folder
    source_files = os.listdir(source_dir)
    data_split_container = create_train_test_val_files(source_files)
    # Create target dir and the folders....
    information_container = create_datagen_dir(target_dir, type_list=('test', 'validation', 'train'), data_list=data_list)
    # Now loop over the created folders
    for data_types, target_type in information_container.items():
        for i_target_type, i_path in target_type.items():
            if i_target_type == sel_target_type:
                for x in data_split_container[data_types]:
                    source_file = os.path.join(source_dir, x)
                    target_file = os.path.join(target_dir, data_types, i_target_type, x)
                    shutil.copy(source_file, target_file)


def create_and_move_data_split(source_dir, target_dir, sel_target_type='input', data_list=None):
    if data_list is None:
        data_list = ('input', 'target')
    # Source_dir is the location that contains all the files
    # These will be split according to 70%, 10%, 20%
    # and put under the target_dir train/test/val folder
    source_files = os.listdir(source_dir)
    data_split_container = create_train_test_val_files(source_files)
    # Create target dir and the folders....
    information_container = create_datagen_dir(target_dir, type_list=('test', 'validation', 'train'), data_list=data_list)
    # Now loop over the created folders
    for data_types, target_type in information_container.items():
        for i_target_type, i_path in target_type.items():
            if i_target_type == sel_target_type:
                for x in data_split_container[data_types]:
                    source_file = os.path.join(source_dir, x)
                    target_file = os.path.join(target_dir, data_types, i_target_type, x)
                    shutil.move(source_file, target_file)


def overrides(interface_class):
    """
    Used to in parent/child methods to prevent stuff being overwritten. See StackOverflow link

    https://stackoverflow.com/questions/1167617/in-python-how-do-i-indicate-im-overriding-a-method
    :param interface_class:
    :return:
    """

    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


def get_square(x):
    """
    Used to get an approximation of the square of a number.
    Needed to place N plots on a equal sized grid.
    :param x:
    :return:
    """
    x_div = simple_div(x)
    x_sqrt = math.sqrt(x)
    diff_list = [abs(y - x_sqrt) for y in x_div]
    res = diff_list.index(min(diff_list))
    # Return largest first...
    return sorted([x_div[res], x // x_div[res]])[::-1]


def type2list(x):
    """
    Converts almost anything to a list.
    Needed when building a configuration dictionary for model runs. To be sure that we can iterate over the object
    There could be something smarter.. but doing list(x) when x is a string.. gives a TypeError
    :param x:
    :return:
    """
    if isinstance(x, list):
        return x
    elif isinstance(x, int):
        return [x]
    elif isinstance(x, str):
        return [x]
    elif isinstance(x, float):
        return [x]
    else:
        warnings.warn('type2list expected list, int, str float. But receveived: {} - {}'.format(type(x), str(x)))


def listdict2dictlist(list_dict):
    # Transforms a list of dictionaries to a dictionary of lists
    set_of_keys = set(itertools.chain(*[list(x.keys()) for x in list_dict]))
    temp_dict = {k: [] for k in set_of_keys}
    for i_key in set_of_keys:
        temp_list = [[v for k, v in x_dict.items() if k == i_key] for x_dict in list_dict]
        temp_list = list(itertools.chain(*temp_list))
        temp_dict[i_key].extend(temp_list)

    return temp_dict


def dictlist2listdict(dict_list):
    # Transforms a dictionary of lists to a list of dictionaries
    return [dict(zip(dict_list, t)) for t in zip(*dict_list.values())]


def simple_generator(list):
    return (x for x in list)


def change_list_order(list_format):
    # Change from [['a', 'b', 'c'], ['a', 'b', 'c']]
    # to [['a', 'a'], ['b', 'b'], ['c', 'c']]
    # and vica versa
    new_format = [[] for _ in range(len(list_format[0]))]
    for i in list_format:
        for ii, j in enumerate(i):
            new_format[ii].append(j)
            # print(f'{ii}, {j} \t {new_format}')
    return new_format


def change_dict_order(x):
    # SImilar to list_order
    # Change from {'a': {'a1': .., 'b1': ..}, 'b': {'a1': .., 'b1': ..}, 'c': {'a1': .., 'b1': ..}}
    # to {'a1': {'a': .., 'b': .. ...}, 'b1': {'a': .., 'b': .. ...}}
    # and vica versa
    new_summary = {}
    for k, v in x.items():
        for ik, iv in v.items():
            new_summary.setdefault(ik, {})
            new_summary[ik][k] = iv
    return new_summary


def update_nested(dict_a, dict_b, debug=False):
    """
    Used to update a nested dictionary with another nested dictionary

    If a value is None, that means it wont get processed. Normally in the code,
    None will go to a default value.
    :param dict_a:
    :param dict_b:
    :param debug:
    :return:
    """
    set_keys = set(dict_a.keys()).union(set(dict_b.keys()))
    if debug:
        print('# # #', set_keys)
    for k in set_keys:
        if debug:
            print(k)
        v = dict_a.get(k)
        if isinstance(v, dict):
            new_dict = dict_b.get(k, None)
            if new_dict:
                update_nested(v, new_dict, debug=debug)
        else:
            new_value = dict_b.get(k, None)
            if new_value is not None:
                dict_a[k] = new_value


def get_nested(dict, keys):
    # Given a list of keys, returns that value of the dict down that path.
    # Example:
    #   temp_dict = {'a': 2, 'b': {'c': {'d': 5}}}
    #   key_path = ['b', 'c', 'd']
    #   get_nested(temp_dict, key_path)

    res = dict
    for key in keys:
        res = res.get(key)
    return res


def set_nested(dic, keys, value):
    # Given a list of keys, sets that value of the dict down that path.
    # Example:
    #   temp_dict = {'a': 2, 'b': {'c': {'d': 5}}}
    #   key_path = ['b', 'c', 'd']
    #   x = -1
    #   hmisc.set_nested(temp_dict, key_path, x)

    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def print_dict(d, level=0, sep_key=':'):
    # Prints a dictionary
    for k, v in d.items():
        if isinstance(v, dict):
            print(level * '\t', k, sep_key)
            print_dict(v, level + 1)
        else:
            print(level * '\t', k, sep_key, v)


def print_dict_mean_value(d, level=0, print_value_dist=20):
    # Prints a dictionary
    for k, v in d.items():
        if isinstance(v, dict):
            print(level * '\t', k, ':')
            print_dict_mean_value(v, level + 1)
        else:
            v = np.array(v)
            v[np.isinf(v)] = 0
            white_space_string = (print_value_dist - len(k)) * ' '
            print(level * '\t', k, ':', white_space_string, np.mean(v))


def print_dict_collection(d, level=0, print_value_dist=20):
    # Prints a dictionary
    for k, v in d.items():
        if isinstance(v, dict):
            print(level * '\t', k, ':')
            print_dict_collection(v, level + 1)
        else:
            v = np.array(v)
            v[np.isinf(v)] = 0
            white_space_string = (print_value_dist - len(k)) * ' '
            print('\n' + level * '\t', k, ':')
            print_dict(collections.Counter(v), sep_key='x')


def write_template(base_template, update_template, dest_dir, target_name, **kwargs):
    # Used to write templates of configuration files for model runs
    debug = kwargs.get('debug', None)
    # Update base template with template
    template_config = copy.deepcopy(base_template)
    update_nested(template_config, update_template, debug=debug)
    ser_template_config = json.dumps(template_config)

    # Pretty print the dictionary to check visually
    if debug:
        print('BASE TEMPLATE')
        print_dict(base_template)
        print('UPDATE TEMPLATE')
        print_dict(update_template)
        print('RESULT CONFIG')
        print_dict(template_config)

    # Write the template
    with open(os.path.join(dest_dir, target_name), 'w') as f:
        f.write(ser_template_config)


def filter_dict_list(x_dict, x_key, sel_value, verbose=False):
    # Get the result config for the chosen patient id..
    sel_x_dict = [x for x in x_dict if x[x_key] == sel_value]
    if len(sel_x_dict) > 1:
        if verbose:
            print(f'We have more than one result. Given value {sel_value}')
        sel_result_dict = sel_x_dict
    elif len(sel_x_dict) == 0:
        if verbose:
            print(f'We have found no results at all. Given value {sel_value}')
        sel_result_dict = 999
    else:
        sel_result_dict = sel_x_dict[0]

    return sel_result_dict


def get_subset_sum(y, subset_sum, epsilon=1, max_iter=1000, method='compare',
                   input_type='count',
                   return_intermediate_results=False):
    # Simple function that (tries to) returns a subset that satisfies the subset_sum argument.
    # Might convergence.. might not.. it is random.
    # 'Compare' methods checks the target value, and changes probabilities based on that
    # 'Random' method is just a little stupid, and has fixed probabilities for changes in the final solution

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if input_type == 'count':
        # In this case we want to subselect with a boolean array to get a subsetsum
        # [12, 5, 2, 1, 6]
        x_init = np.zeros(len(y), dtype=bool)
    elif input_type == 'balance':
        # Here we want to select the index in an array, becuase we can choose between positive or negative
        # [(5, -2), (3, -1), ...]
        # It might be that we dont want this case anymore..
        x_init = np.zeros(len(y), dtype=int)
    else:
        print('Unknown input_type ', input_type)
        return None

    x_result = np.copy(x_init)

    temp_result = y[x_result].sum()

    counter = 0
    storage_results = []
    while np.abs(temp_result - subset_sum) > epsilon and counter < max_iter:
        if method == 'random':
            p_true = 0.6
        elif method == 'compare':
            if temp_result - subset_sum > 0:
                p_true = 0.9
            else:
                p_true = 0.1
        else:
            p_true = 0

        # Random on
        rand_on_ind = np.random.choice(len(y))
        if np.random.rand() > p_true:
            x_result[rand_on_ind] = True

        # Random off
        rand_on_ind = np.random.choice(len(y))
        if np.random.rand() > (1 - p_true):
            x_result[rand_on_ind] = False

        if input_type == 'count':
            interm_solution = y[x_result]
        elif input_type == 'balance':
            interm_solution = np.array([y_count[x_ind] for y_count, x_ind in zip(y, x_result)])
        else:
            interm_solution = None

        temp_result = interm_solution.sum()
        storage_results.append(temp_result)
        counter += 1

    bool_convergence = counter < max_iter
    print('Reached convergence ', bool_convergence)
    print('Convergence value ', temp_result)
    print('Target value ', subset_sum)
    print('Number of interations ', counter)

    if return_intermediate_results:
        return x_result, storage_results
    else:
        return x_result


def dice_metric(x, y):
    smooth = 1.
    iflat = x.ravel()
    tflat = y.ravel()
    intersection = (iflat * tflat).sum()
    A_sum = (iflat * iflat).sum()
    B_sum = sum(tflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


def move_list_of_files(orig, dest, files):
    [shutil.move(os.path.join(orig, x), dest) for x in files]
    print('Move is done')


def get_coil_position(input_array_abs):
    # inp_abs = np.sqrt(input_array[::2] ** 2 + input_array[1::2] ** 2)
    inp_abs = input_array_abs
    inp_norm = inp_abs / (np.linalg.norm(inp_abs, axis=0) + 1e-5)
    inp_norm[np.isnan(inp_norm)] = 0
    inp_norm = np.abs(inp_norm)
    inp_norm_max = np.max(inp_norm, axis=(1, 2))
    max_position_ind = []
    for i in range(len(inp_abs)):
        pos_y, pos_x = map(int, np.where(inp_norm[i] == inp_norm_max[i]))
        # print('Max value ', inp_norm[i][pos_y, pos_x])
        max_position_ind.append([pos_y, pos_x])

    return max_position_ind


def get_line_length(x):
    # Assumes equally spaced stuff
    line_length = np.sqrt(np.diff(x) ** 2 + 1).sum()
    return line_length


def get_maximum_curvature(x_coord, y_coord):
    line_coords = np.stack([x_coord, y_coord])
    curvature = get_curvature(line_coords.T)
    index_max = np.argmax(curvature)
    return curvature, index_max

def get_curvature(x):
    # From a set of 2-D points in X
    # https: // en.wikipedia.org / wiki / Curvature
    dx_dt = np.gradient(x[:, 0])
    dy_dt = np.gradient(x[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt ** 2 + dy_dt ** 2) ** 1.5
    return curvature


def convert2tensor(x):
    import torch
    x = harray.scale_minmax(np.abs(x))
    if x.ndim == 2:
        x = x[None, None]
    elif x.ndim == 3:
        x = x[None]
    else:
        print('Dimension of x ', x.ndim)

    x_tens = torch.as_tensor(x).float()
    return x_tens


def convert2tensor_complex(x, stack_ax=0):
    import torch
    if x.ndim == 3:
        n_y, n_x = x.shape[-2:]
        x_stacked = harray.to_stacked(x, cpx_type='cartesian', stack_ax=stack_ax)
        x_stacked = x_stacked.T.reshape((n_x, n_y, -1)).T
        x_tensor = torch.as_tensor(x_stacked[np.newaxis]).float()
    else:
        print('Sorry, dimension is wrong. Currently: ', x.ndim)
        x_tensor = None

    return x_tensor

def get_freedman_bins(x):
    # Calculate the Freedman–Diaconis_rule
    # Source: https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    n = len(x)
    iqr_manual = np.quantile(x, q=[.25, .75])
    IQR = np.diff(iqr_manual)[0]
    if len(set(x)) == 1 or IQR == 0:
        return None
    # Using Friedman Diaconis rule...
    bin_width = 2 * IQR / n ** (1 / 3)
    bins = np.arange(min(x), max(x), bin_width)
    return bins


def load_array(input_file, data_key='data', convert2gray=False, sel_slice=None):
    ext = get_ext(input_file)
    base_name = get_base_name(input_file)
    if 'h5' in ext:
        with h5py.File(input_file, 'r') as h5_obj:
            if data_key in h5_obj.keys():
                if sel_slice is None:
                    loaded_array = np.array(h5_obj[data_key])
                elif sel_slice == 'mid':
                    mid_slice = h5_obj[data_key].shape[0] // 2
                    loaded_array = np.array(h5_obj[data_key][mid_slice])
                else:
                    loaded_array = np.array(h5_obj[data_key][sel_slice])
            else:
                loaded_array = None
                warn_str = f"Unknown key {data_key}. Accepting keys {h5_obj.keys()}"
                warnings.warn(warn_str)
    elif 'npy' in ext:
        loaded_array = np.load(input_file)
    # Scary om met IM_ te checken...
    elif ('dicom' in ext) or ('dcm' in ext) or (base_name.startswith('IM_')):
        loaded_array = pydicom.read_file(input_file).pixel_array
    elif 'nii' in ext:
        loaded_array = nibabel.load(input_file).get_fdata()
    elif ('png' in ext) or ('jpg' in ext) or ('jpeg' in ext):
        pillow_obj = Image.open(input_file)
        if convert2gray:
            loaded_array = np.array(pillow_obj.convert('LA'))
        else:
            loaded_array = np.array(pillow_obj.convert("RGB"))
    elif 'mat' in ext:
        mat_obj = scipy.io.loadmat(input_file)
        # Filter out all the protected names
        if data_key in mat_obj.keys():
            loaded_array = mat_obj[data_key]
        else:
            print('Unknown key in mat obj: ', mat_obj.keys())
            print("File name ", input_file)
            print("Returning matlab object")
            loaded_array = mat_obj
    else:
        print('Unknown extension ', input_file, ext)
        loaded_array = np.array(None)

    return loaded_array


def check_and_create_dir(dir):
    if isinstance(dir, str):
        if not os.path.isdir(dir):
            os.makedirs(dir)
    elif isinstance(dir, list):
        for i_dir in dir:
            check_and_create_dir(i_dir)


def find_index_file(file_list, target_file):
    base_name = get_base_name(target_file)
    file_list = [get_base_name(x) for x in file_list]
    index = file_list.index(base_name)
    return index

# Useful to test functionality of functions
# from timeit import Timer
# def foo():
#     return [np.random.randint(10) for i in range(18)]
#
# def foobar():
#     return np.random.randint(10, size=(18,))
#
# t1 = Timer("""foo()""", """from __main__ import foo""")
# t2 = Timer("""foobar()""", """from __main__ import foobar""")
# t1.timeit(5000)  # runs foo() 1000 times and returns the time taken
# t2.timeit(5000)


class PrintFileShape:
    def __init__(self, ddir):
        # Using `ext in x` because, for example, most DICOMS are of the name IM0002.
        self.file_list = []
        for i_file in os.listdir(ddir):
            temp = os.path.join(ddir, i_file)
            if os.path.isfile(temp):
                self.file_list.append(temp)

    def print_file_shape(self, data_key=None):
        shape_list = []
        for i_file in sorted(self.file_list):
            ext = get_ext(i_file)
            file_name = get_base_name(i_file)
            temp = load_array(i_file, data_key=data_key)
            temp_shape = temp.shape
            shape_list.append(temp_shape)
            print(f'Read file {file_name} - shape {temp_shape} - file extension {ext}')

        return collections.Counter(shape_list)

    def print_file_affine(self):
        affine_list = []
        for i_file in sorted(self.file_list):
            ext = get_ext(i_file)
            file_name = get_base_name(i_file)
            if ext == '.nii.gz':
                nibabel_obj = nibabel.load(i_file)
                affine_struct = nibabel_obj.affine
                affine_list.append(affine_struct)
                print(f'Read file {file_name} - affine {affine_struct}')

        return collections.Counter(affine_list)


def circular_mask(img_shape):
    x_size, y_size = img_shape
    x_range = np.linspace(-x_size // 2, x_size // 2, x_size)
    y_range = np.linspace(-y_size // 2, y_size // 2, y_size)
    X, Y = np.meshgrid(x_range, y_range)
    mask_array = np.sqrt(X ** 2 + Y ** 2) <= x_size // 2
    return mask_array