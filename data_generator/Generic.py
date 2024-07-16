# encoding: utf-8

import helper.array_transf as harray
import helper.misc as hmisc
import numpy as np
import torch.utils.data
import os
import random
import re
import inspect
import random
import h5py
import itertools
import skimage.transform as sktransform
import nibabel

import torchio
import inspect
import helper_torch.torchio_transforms as htransforms
"""
Here we define generic components of the Data Generotr DataSet
"""


def transform_array(x, transform_type, complex_type='cartesian', stack_ax=-1, concatenate_complex=True):
    if transform_type == 'complex':
        x_shape = x.shape
        im_y, im_x = x_shape[-2:]  # Get the last two indices
        # If have a simple 2D image.. add one axes..?
        if x.ndim == 2:
            x = x[np.newaxis]

        # Split the complex parts into a new axis (stack_ax)
        x = harray.to_stacked(x, cpx_type=complex_type, stack_ax=stack_ax)
        if concatenate_complex:
            # Return a (2 * channel, x, y) image
            # IF False.. then we have an extra dimension where the complex parts are
            x = x.T.reshape((im_x, im_y, -1)).T
    elif transform_type == 'sumcomplex':
        x = np.abs(x.sum(axis=0, keepdims=True))
    elif transform_type == 'abssumcomplex':
        x = np.abs(x).sum(axis=0, keepdims=True)
    elif transform_type == 'abs':
        x = np.abs(x)
    elif transform_type == 'angle':
        x = np.angle(x)
    elif transform_type == 'cos_angle':
        x = np.cos(np.angle(x))
    elif transform_type == 'real':
        x = np.real(x)
    elif transform_type == 'real_norm':
        x = harray.scale_minmax(np.real(x), axis=(1, 2))
    elif transform_type == 'imag':
        x = np.imag(x)
    elif transform_type == 'cos':
        # This one is a little bit outdated..
        x = harray.to_stacked(np.cos(x), cpx_type=complex_type, stack_ax=stack_ax)
    elif transform_type == 'arcsinh':
        x = np.arcsinh(x)
    else:
        print('Unkown transform_type: ', transform_type)

    return x


class DatasetGeneric(torch.utils.data.Dataset):
    """
    Basic dataset generator. Contains all the possible options to tune your data generator
    """
    def __init__(self, ddata, input_shape=None, target_shape=None, shuffle=True, dataset_type='train', transform=None, file_ext='npy', **kwargs):
        """
        :param ddata: Can be a concattenation of multiple source folders. Seperated by a ;
        :param input_shape: Not always needed. But can be useful when doing data transformations
        :param target_shape: Not always needed. Sometimes used for data augmentation
        :param shuffle: Whether or not ot shuffle the file list
        :param dataset_type: train/test/validation
        :param transform:
        :param file_ext:
        :param kwargs:

        'input_is_output' - Uses the same file for target array as input array
        'filter_string' - Used to filter items from the list of data files
        'optional_input_appendix' - Additional string to change input folder
        'optional target appendix' - Additional string to change target folder
        """
        # Variation options
        self.input_is_output = kwargs.get('input_is_output', False)
        self.output_is_input = kwargs.get('output_is_input', False)
        self.switch_input_output = kwargs.get('switch_input_output', False)

        self.number_of_examples = kwargs.get('number_of_examples', 9999999)
        self.filter_string = kwargs.get('filter_string', None)
        self.masked = kwargs.get('masked', False)  # It is used
        # This is used to cycle through all the items in the TEST set
        # Even though we have a single file with multiple slices....
        self.cycle_all_items = kwargs.get('cycle_all', False)
        self.center_slice = kwargs.get('center_slice', False)
        self.optional_input_appendix = kwargs.get('optional_input_appendix', '')
        self.optional_target_appendix = kwargs.get('optional_target_appendix', '')
        self.debug = kwargs.get('debug', None)
        self.debug_cuda = kwargs.get('debug_cuda', False)

        self.transform_resize = kwargs.get('transform_resize', False)
        self.resize_list = [(64, 64), (128, 128), (256, 256), (512, 512)]
        self.resize_index = len(self.resize_list) - 1
        self.file_ext = file_ext

        self.shuffle = shuffle
        self.dataset_type = dataset_type
        self.container_file_info = []
        self.debug_display_counter = 0
        ddata_list = ddata.split(';')

        # This way we can deal with multiple input paths...
        for i_ddata in ddata_list:
            temp_dict = {}
            if i_ddata:
                input_dir, target_dir, mask_dir = self.get_load_dir(i_ddata)
                file_list = [x for x in os.listdir(input_dir) if x.endswith(file_ext)]
                # Randomly select k samples. But check with file size if possible#
                n_max = len(file_list)
                sel_files = np.min([n_max, self.number_of_examples])
                file_list = random.sample(file_list, k=sel_files)

                slice_count = {}
                # Cycle through all the items when we are dealing with a test set
                if self.dataset_type == 'test' and self.cycle_all_items:
                    for i_mask_file in file_list:
                        temp_mask_file = os.path.join(mask_dir, i_mask_file)
                        # For now only h5 is relevant...
                        if 'h5' in file_ext:
                            h5_file = h5py.File(temp_mask_file, 'r')
                            max_slice = h5_file['data'].shape[0]
                            # Store the full range of a single file so we can `pop` it later during evaluation
                            slice_count[i_mask_file] = list(set([int(x) for x in np.linspace(0, max_slice-1, 25)]))

                    # Now each element of the file is repeated `max_slice` times.
                    file_list = [[x] * len(slice_count[x]) for x in file_list]
                    file_list = list(itertools.chain(*file_list))

                temp_dict['input_dir'] = input_dir
                temp_dict['target_dir'] = target_dir
                temp_dict['mask_dir'] = mask_dir
                temp_dict['file_list'] = sorted(file_list)
                temp_dict['slice_count'] = slice_count
            else:
                # Here we process the case when we just have an empty input dir
                # This could happen when we just generate random input/target files
                temp_dict['input_dir'] = '/'
                temp_dict['target_dir'] = '/'
                temp_dict['mask_dir'] = '/'
                temp_dict['file_list'] = ['' for _ in range(self.number_of_examples)]

            self.container_file_info.append(temp_dict)

        if self.filter_string is not None:
            re_filter = re.compile(self.filter_string)
            for x in self.container_file_info:
                x['file_list'] = sorted([i_file for i_file in x['file_list'] if re_filter.match(i_file)])

        self.n_datasets = len(ddata_list)
        # This still changes the interal lists
        if self.shuffle:
            [random.shuffle(x['file_list']) for x in self.container_file_info]

        # Check the image shape...
        self.img_input_shape = input_shape
        if target_shape is None:
            self.img_target_shape = input_shape
        else:
            self.img_target_shape = target_shape

        self.transform_compose = None
        if transform is not None:
            transform_dict = dict(inspect.getmembers(torchio.transforms, inspect.isclass))
            # htransform_dict = dict(inspect.getmembers(htransforms, inspect.isclass))
            # transform_dict.update(htransform_dict)
            transform_list = []
            for transform_name, transform_config in transform.items():
                transform_obj = transform_dict.get(transform_name, None)(**transform_config)
                transform_list.append(transform_obj)

            self.transform_compose = torchio.transforms.OneOf(transform_list)

            print('Using transforms: ')
            hmisc.print_dict(transform)

        if self.debug:
            print(f'\nDatasetGeneric - {dataset_type} - init parameters:')
            print('\t\t Total number of items ', [len(x['file_list']) for x in self.container_file_info])
            print('\t\t Example of 10 items ', [x['file_list'][:10] for x in self.container_file_info])
            print('\t\t Length of data generator ', self.__len__())
            print('\t\t Input path ', [x['input_dir'] for x in self.container_file_info])
            print('\t\t Target path ', [x['target_dir'] for x in self.container_file_info])
            print('\t\t Loaded image shape ', self.img_input_shape, self.img_target_shape)
            print('\t\t Input is output ', self.input_is_output)
            print('\t\t Output is input ', self.output_is_input)
            print('\t\t Number of examples ', self.number_of_examples)
            print('\t\t Filtering on input ', self.filter_string)
            print('\t\t Optional input appendix ', self.optional_input_appendix)
            print('\t\t Optional target appendix ', self.optional_target_appendix)

    def __len__(self):
        n_files = sum([len(x['file_list']) for x in self.container_file_info])
        return n_files

    def get_load_dir(self, ddata):
        if self.debug:
            print('DatasetGeneric - set load dir')

        if self.dataset_type == 'train':
            temp_dir = os.path.join(ddata, 'train')
        elif self.dataset_type == 'validation':
            temp_dir = os.path.join(ddata, 'validation')
        elif self.dataset_type == 'test':
            temp_dir = os.path.join(ddata, 'test')
        else:
            # temp_dir = '~'
            temp_dir = ddata
            print('Dataset unknown dataset type: ', self.dataset_type)

        # Since we CAN define multiple input 'bases'..
        input_dir = os.path.join(temp_dir, 'input' + self.optional_input_appendix)
        target_dir = os.path.join(temp_dir, 'target' + self.optional_target_appendix)
        mask_dir = os.path.join(temp_dir, 'mask')

        if self.input_is_output:
            print('Target image is equal to input, and input is input')
            target_dir = input_dir

        elif self.output_is_input:
            print('Input image is equal to target, and target is target')
            input_dir = target_dir
        elif self.switch_input_output:
            print('Switching input and output')
            temp = target_dir
            target_dir = input_dir
            input_dir = temp

        if self.debug:
            print('\t Using image paths ', input_dir)
            print('\t                   ', target_dir)

        return input_dir, target_dir, mask_dir

    def set_resize_parameters(self):
        """
        We want to be able to set certain transformation parameters over each batch...
        Like resize transformations etc..

        With this we are able to do that
        :return:
        """
        self.resize_index = np.random.randint(0, len(self.resize_list))

    @staticmethod
    def perform_resize(x, resize_shape):
        orig_shape = x.shape[:-2]
        new_shape = orig_shape + resize_shape
        x = sktransform.resize(x, new_shape, preserve_range=True, anti_aliasing=False)
        return x

    def on_epoch_end(self):
        'Updates file_list after each epoch'
        if self.shuffle:
            np.random.shuffle(self.file_list)

    def print(self):
        hmisc.print_dict(self.__dict__)


class DatasetGenericComplex(DatasetGeneric):
    """
    Addition to the Generic Dataset creator to allow for complex transformations
    """
    def __init__(self, ddata, input_shape=None, target_shape=None, shuffle=True, dataset_type='train', file_ext='npy', **kwargs):
        """
        :param ddata:
        :param input_shape:
        :param target_shape:
        :param shuffle:
        :param dataset_type:
        :param file_ext:
        :param kwargs:

        'complex_type' - type of complex values that are returned. Cartesian or Polar
        'transform_type_input' - data augmentation for input and output (of _target is not given)
        'transform_type_target' - specific augmentation for target
        'stack_ax' - axis on which the complex parts are being stacked
        'concatenate_complex' - if True [default] transform 8 complex coils to 16 channels. If false, transforms to (2, 8) channels
        """
        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, **kwargs)

        self.complex_type = kwargs.get('complex_type', 'cartesian')
        self.transform_type_input = kwargs.get('transform_type', 'complex')
        self.transform_type_target = kwargs.get('transform_type_target', None)
        self.stack_ax = kwargs.get('stack_ax', 0)
        self.concatenate_complex = kwargs.get('concatenate_complex', True)

        # These are defined to have 8 coils as output (real valued) opposed to 16.
        self.transform_exceptions = ['real', 'imag', 'abs', 'angle', 'cos_angle']

        if self.debug:
            print('\nStatus of kwargs Dataset Generic Complex:')
            print('\t\t Complex type ', self.complex_type)
            print('\t\t Transform type ', self.transform_type_input)
            print('\t\t Transform type target', self.transform_type_target)
            print('\t\t Stack axis ', self.stack_ax)
            print('\t\t Concat complex ', self.concatenate_complex)

    def transform_complex(self, x, transform_type=None, stack_ax=-1, complex_type=None):
        """
        Here we transform a varialbe x towards a specific type...
        This also converts complex-valued arrays to stacked real/imag valued ones

        :param x:
        :param transform_type:
        :param stack_ax:
        :return:
        """
        if transform_type is None:
            transform_type = self.transform_type_input

        if complex_type is None:
            complex_type = self.complex_type

        if x.ndim == 2:
            x = np.expand_dims(x, stack_ax)

        x = transform_array(x=x, transform_type=transform_type, complex_type=complex_type,
                        stack_ax=self.stack_ax, concatenate_complex=self.concatenate_complex)

        return x