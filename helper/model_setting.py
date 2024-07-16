# encoding: utf-8

"""
Helper functions to set model settings..
"""

# Standard code
import json
import os
import copy
import itertools
import shutil

# Self created code
import helper.misc as hmisc


def create_mult_dict(compact_dict, packed_keys, **kwargs):
    # Here we create a function that creates multiple dictionaries using the cartesian product on the keys provided
    # by expand_keys.

    # Check which keys are in the first layers of keys of compact dict
    # You can easily add those to the list of combinations

    # Find remainder of keys in nested dict
    # Eventually add those values to the combinations as well...

    # This way we can deal with nested dictionaries as well. Does increase the complexity. So I might revert back to
    # having it on one level instead.
    debug = kwargs.get('debug', None)

    # If we have not set a output directory, set it the same as the template dir
    if compact_dict['dir'].get('doutput', None) is None:
        compact_dict['dir']['doutput'] = compact_dict['dir']['dtemplate']

    packed_keys = [hmisc.type2list(x) for x in packed_keys]
    if debug:
        print('Possible keys')
        for i_key in packed_keys:
            print(i_key)

    res_values = [hmisc.type2list(hmisc.get_nested(compact_dict, x)) for x in packed_keys]
    config_iter = itertools.product(*res_values)

    mult_dict = {}
    for i, i_config in enumerate(config_iter):
        temp = copy.deepcopy(compact_dict)
        [hmisc.set_nested(temp, x, y) for x, y in zip(packed_keys, i_config)]
        if debug:
            hmisc.print_dict(temp)
        mult_dict.update({'config_{:02d}'.format(i): temp})

    return mult_dict


def _find_kspace_pairs(mult_dict, **kwargs):
    """
    Among all the dictionaries (which are of the form.. {'config_01': {...}, 'config_02': {...}, ...}),
    find the keys ('config_01', ...) which only differ in 'img_type'.
    Those that satify the combination ('real', 'imag') or ('abs', 'angle') are put together.

    :param mult_dict:  a result of create_mult_dict
    :return: a list of tuples that gives the duo-config names that are k-space related.
    """
    debug = kwargs.get('debug')

    res = []

    for x_config, x_dict in mult_dict.items():
        if debug:
            print('First config ', x_config, x_dict['data']['img_type'])

        temp_dict = copy.deepcopy(x_dict)
        # hard coded location of img_typ
        img_type = temp_dict['data']['img_type']
        if img_type == 'real':
            temp_dict['data']['img_type'] = 'imag'
        elif img_type == 'abs':
            temp_dict['data']['img_type'] = 'angle'
        else:
            pass

        for y_config, y_dict in mult_dict.items():
            if y_config != x_config:
                if debug:
                    print('Second config ', y_config, y_dict['data']['img_type'])
                if temp_dict == y_dict:
                    res.append((x_config, y_config))
                    if debug:
                        print('Found combi ', x_config, y_config)

    return res


def create_kspace_config_dir(model_path, kspace_combi_dict, config_name='config_param.json'):
    # Creates and moves the necessary files and folders to accomodate the recombination of kspace data predictions

    res = []
    if kspace_combi_dict:
        for i, i_combi in enumerate(kspace_combi_dict):
            for i_config in i_combi:
                old_file = os.path.join(model_path, i_config, config_name)
                new_path = os.path.join(model_path, 'config_kspace_' + str(i).zfill(2))
                file_name = str(i_config) + '.json'
                new_file = os.path.join(new_path, file_name)
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)
                res.append(new_file)
                shutil.copyfile(old_file, new_file)
    return res


def create_config_dir(output_path, multi_dict, config_name='config_param.json', **kwargs):
    """

    :param output_path:
    :param multi_dict:
    :param name:
    :param kwargs:
    :return: list of all the model paths
    """

    debug = kwargs.get('debug', None)

    model_path_list = []

    # Clean out the whole directory... EXCEPT the config_run.json file..
    for root, dirs, files in os.walk(output_path):
        files = [x for x in files if 'config_run.json' not in x]
        if debug:
            print('Deleting', root, dirs, files)
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    # Loop over the (nested) dictionary and create configs for each model run
    for i_config in multi_dict.keys():
        full_output_path = os.path.join(output_path, i_config)
        temp_config = multi_dict[i_config]

        if not os.path.isdir(full_output_path):
            os.makedirs(full_output_path)
        if debug:
            print('Created directory ', full_output_path)

        # This is for the config temp. It should be different from the main json template
        temp_config['dir']['doutput'] = full_output_path

        # Write temp config
        ser_json_config = json.dumps(temp_config)
        temp_config_name = os.path.join(full_output_path, config_name)
        with open(temp_config_name, 'w') as f:
            f.write(ser_json_config)


        model_path_list.append(full_output_path)

    return model_path_list

