
"""
Here we define the class that handles are image extraction processes
"""

import warnings
import time
import inspect
import os
import re
import itertools
import numpy as np
import pandas as pd
import skimage.transform as sktransf
import tabulate
from operator import itemgetter


class ListReader:
    """
    To read the .list file, parse the header content and put the off-set table in a pandas DataFrame
    """
    def __init__(self, input_file, **kwargs):
        self.input_file = input_file
        self.debug = kwargs.get('debug', False)
        # Not needed for parsing the list-file
        self.scan_name = self.get_scan_name()

    def get_scan_name(self, input_file=None, name_string=None):
        """
        Simple function to get the filename from files .list, .par, .sin
        """
        if input_file is None:
            input_file = self.input_file

        input_file_no_ext, ext = os.path.splitext(input_file)

        # Update ext with file_ext, if that one is not none or empty
        n_lines = 14
        if ext == '.list':
            name_string = name_string or 'name'
            ind_name = 1
        elif ext == '.par':
            name_string = name_string or 'Protocol name'
            ind_name = 1
        elif ext == '.sin':
            name_string = name_string or 'scan_name'
            ind_name = 2
        else:
            warnings.warn('Unidentified extension {}\nReading zero lines'.format(ext))
            n_lines = 0
            ind_name = 0
            name_string = ''

        with open(input_file, 'r') as f:
            description_file = [f.readline() for _ in range(n_lines)]

        # Content is separated by a colon.
        file_name = [x.split(':')[ind_name].strip() for x in description_file if name_string in x]

        # Prevent errors if we havent found anything
        if len(file_name):
            file_name = file_name[0]
        else:
            file_name = "None found"

        return file_name

    def get_list_file(self, input_file=None):
        """
        Parses the list file and return the reference table for the .data file and the parameter table
        """
        if input_file is None:
            input_file = self.input_file

        # Make sure that we have a .list file
        file_no_ext, _ = os.path.splitext(input_file)
        input_file = file_no_ext + '.list'
        # Predefined content of the .list file to locate where important information is.
        start_param_criteria = '# === GENERAL INFORMATION ========================================================\n'
        start_data_criteria = '# === START OF DATA VECTOR INDEX =================================================\n'
        end_data_criteria = '# === END OF DATA VECTOR INDEX ===================================================\n'
        end_file_criteria = '# === END OF DATA DESCRIPTION FILE ===============================================\n'

        with open(input_file, 'r') as f:
            header_list_file = f.readlines()

        # Get indices where each part of the data is
        ind_start_param = header_list_file.index(start_param_criteria)
        ind_start_data = header_list_file.index(start_data_criteria)
        ind_end_data = header_list_file.index(end_data_criteria)
        ind_end_file = header_list_file.index(end_file_criteria)

        # Since parameters are both in the header and trailer of the file, we concat them here
        list_param_header = header_list_file[ind_start_param:ind_start_data]
        list_param_trailer = header_list_file[ind_end_data:ind_end_file]
        parameter_list = list_param_header + list_param_trailer
        parameter_table = self._parse_parameter_list(parameter_list)

        # The reference data is in a simple structured format. Here we extract it to a pandas dataframe
        reference_data_list = header_list_file[ind_start_data:ind_end_data]
        reference_table = self._parse_reference_list(reference_data_list)
        reference_table = self._set_dtype_ref_data(reference_table)

        return reference_table, parameter_table

    def _parse_parameter_list(self, parameter_list):
        """
        Parse the variables mentioned in the .list file.
        Returns a pandas dataframe, which functions like a dictionary with multi index.
        """
        # Reasoning behind regex.. An example file-line:
        # .    0    0    0  number_of_extra_attribute_1_values :     1
        re_line = re.compile('^\.\s+([0-9])\s+([0-9])\s+([0-9])\s+(\w+-\w+|\w+)\s+:\s+(.*)')

        # We need to remove certain \#.. Example below belongs to the same 'group'
        # But is not recognized because of the dicontinuity caused by the \#
        # .    0    0    0  number_of_signal_averages          :     1
        # #
        # .    1    0    0  number_of_encoding_dimensions      :     4
        # Find all rows that have a \# to start with, but a \. before and after.
        removed_indices = self._remove_unwanted_breaks(parameter_list, debug=self.debug)
        # We cannot delete those indices directly.. because that would shift the index
        for i_remove in removed_indices:
            parameter_list[i_remove] = ''
        parameter_list = [x for x in parameter_list if x]

        # We want to separate the different configurations for different mix, echo, or loc values.
        # These will be coined the 'id' values.
        temp_list_dict = []
        # Since in the .list file we see that certain column values belong to all mix/echo/loc..
        # Below variables help with identifier them
        temp_group = []
        group_id = []
        combi_list = []
        prev_line = False
        for i, i_line in enumerate(parameter_list):
            if re_line.match(i_line):
                if self.debug:
                    print(f'\n\n\n {i} Parsing the following line:\t"', i_line.strip(), '"')

                if prev_line is False:
                    # Now we have the first match of the group...
                    combi_list.append([group_id, temp_group])
                    temp_group = []
                    # This is needed as prep work for the 'next' line
                    group_id = [x for x in parameter_list[i - 2].split()[1:4] if x != 'n.a.']
                    if self.debug:
                        print('\nGathered group id ', group_id)
                        print('Using... ', parameter_list[i - 2])

                res = re_line.findall(i_line)[0]
                temp_dict = {'id': list(map(int, res[:3])), 'value': self._convert_type(*res[3:])}

                if self.debug:
                    print('Obtaining the following key-value pair\n', temp_dict)

                temp_list_dict.append(temp_dict)
                temp_group.append(res[3])
                prev_line = True
            else:
                prev_line = False

        # We want to group the values of similar 'id's together. To do so, we first need to sort them
        sorted_param_dict = sorted(temp_list_dict, key=itemgetter('id'))

        if self.debug:
            print('\n \t === Regrouping the found dict ===')

        # Here the grouping and extraction takes place.
        # itertools.groupby method is crucial here.
        param_list_dict = []
        col_names = ['mix', 'echo', 'loca']
        for key, group in itertools.groupby(sorted_param_dict, key=lambda x: x['id']):
            temp_dict = dict(zip(col_names, key))
            for group_item in group:
                temp_dict.update(group_item['value'])

            if self.debug:
                print('\n Grouping over group combination ', key)

            param_list_dict.append(temp_dict)

        param_table = pd.DataFrame(param_list_dict).set_index(col_names)

        # Here we propagate the known values to the same group..
        combi_list = [x for x in combi_list if len(x[1])]
        for i_group, i_col_list in combi_list:
            if self.debug:
                print('\nGrouping over ', i_group)
                print('With columns ', i_col_list, end='\n')

            for i_col in i_col_list:
                if i_group:
                    param_table[i_col] = param_table.groupby(i_group)[i_col].ffill().bfill()
                else:
                    param_table[i_col] = param_table[i_col].ffill().bfill()

        return param_table

    @staticmethod
    def _remove_unwanted_breaks(parameter_list, debug=False):
        temp = []
        for i, x in enumerate(parameter_list):
            if i-1 > 0:
                prev_line_start = parameter_list[i-1][0]
            else:
                prev_line_start = '#'
            if i+1 < len(parameter_list):
                next_line_start = parameter_list[i+1][0]
            else:
                next_line_start = '#'

            if x[0] == '#' and prev_line_start == '.' and next_line_start == '.':
                temp.append(i)
                if debug:
                    print(i, prev_line_start, x[0], next_line_start)
        return temp

    # Below are some private functions for this file..
    @staticmethod
    def _parse_reference_list(reference_data):
        """
        Used inside parse_list_file.
        Used to create a table from the reference data in the .list file
        """

        # Column names are put at a specific position. This position is hardcoded here
        column_names = reference_data[2][2:].split()

        # Data is separated by empty spaces, and starts and a predefined position. This position is hardcoded here
        pd_list_reference = pd.DataFrame([x.split() for x in reference_data[5:-1]])
        pd_list_reference.columns = column_names

        return pd_list_reference

    @staticmethod
    def _set_dtype_ref_data(reference_data):
        """
        Used inside parse_list_file
        Convert all columns to int, except typ and offset
        """

        # Obtain all column names
        to_int_col = set(reference_data.columns.values)
        to_int_col.difference_update(['typ', 'offset'])

        # Use a copy of the column to set the new column value.
        for i_col in to_int_col:
            reference_data[i_col] = reference_data[i_col].astype(int)

        return reference_data

    @staticmethod
    def _convert_type(name, value):
        """
        Used inside _parse_parameter_list
        Small function to change the type of the .list file strings to a proper type
        """

        if 'range' in name:
            value = list(map(int, value.split()))
        elif name == 'coil_channel_combination':
            value = str(value)
        else:
            value = float(value)
        return {name: value}


class DataListImage:
    def __init__(self, input_file, **kwargs):
        # post_proc: only needed with image data
        self.debug = kwargs.get('debug', False)
        self.status = kwargs.get('status', None)
        self.complex = kwargs.get('complex', None)  # Used to denote that we want to retrieve the complex numbres.
        # sel_file is the file name without .data or .list extension

        self.input_file = input_file
        self.list_reader_obj = ListReader(input_file)
        self.reference_table, self.parameter_table = self.list_reader_obj.get_list_file()

        # We use these indices to loop over eventually
        # Therefore we set self.data_frame.set_index(self.index_col)
        index_col_image = ['mix', 'echo', 'loca', 'dyn', 'chan', 'card', 'z', 'y']
        index_col_kspace = ['mix', 'echo', 'loca', 'dyn', 'chan', 'card', 'kz', 'ky']
        if 'ky' in self.reference_table.columns:
            self.index_col = index_col_kspace
        else:
            self.index_col = index_col_image

        # Get parameters from param file
        # Here we select (mix=0, echo=0, loc=0).
        n_rows = len(self.parameter_table.index)
        if n_rows > 1:
            warnings.warn('We have multiple mix/echo/loc combinations. Choosing the first one')
            print(self.parameter_table.index)
            self._check_difference(self.parameter_table)

        self.parameter_table = self.parameter_table.loc[(0, 0, 0)]
        self.n_enc = self.parameter_table.get('number_of_encoding_dimensions').astype(int)
        self.n_dyn = self.parameter_table.get('number_of_dynamic_scans').astype(int)
        self.n_loc = self.parameter_table.get('number_of_locations').astype(int)
        self.enc_dim = [-3, -2, -1][-self.n_enc:]

        print('Number of encoding dimensions ', self.n_enc)
        print('Number of dynamics ', self.n_dyn)
        print('Number of locations ', self.n_loc)

    def get_image_data(self, input_file=None, sel_loc=None):
        if input_file is None:
            input_file = self.input_file

        # Select (by default) only STD data
        std_data = self.get_std_data()
        if sel_loc is not None:
            std_data = self.get_loc_data(std_data, sel_loc=sel_loc)

        std_data = std_data.set_index(self.index_col)

        # Obtain the data from the .data file
        data_array = self.get_data_content(ref_data=std_data)
        # Transform it to an image (if needed)
        image_data = self.transform_scan_data(data_array=data_array)
        return data_array, image_data

    def get_noise_data(self, input_file=None):
        # Select Noise data
        NOI_data = self.get_std_data(sel_type='NOI')
        n_noise = np.prod(NOI_data.shape)
        # Obtain the data from the .data file
        if n_noise > 0:
            data_array = self.get_data_content(ref_data=NOI_data)
            return data_array
        else:
            print('No noise data available')


    def get_std_data(self, sel_type='STD'):
        """
        Extracts the STD-labeled rows from the reference table
        :param sel_type:
        :return:
        """
        if self.status:
            print(time.ctime(), ' - ', inspect.stack()[0][3])
        # STD stands for Standard Datavector
        temp_data = self.reference_table.loc[self.reference_table['typ'] == sel_type]
        return temp_data

    def get_loc_data(self, ref_table, sel_loc=0):
        if self.status:
            print(time.ctime(), ' - ', inspect.stack()[0][3])
        # STD stands for Standard Datavector
        if 'loca' in ref_table.columns:
            temp_data = ref_table.loc[ref_table['loca'] == sel_loc]
        elif 'loc' in ref_table.columns:
            temp_data = ref_table.loc[ref_table['loc'] == sel_loc]
        else:
            print('No locatin found')
            temp_data = None

        return temp_data

    def get_data_content(self, ref_data, input_file=None):
        """
        Here we read the .data file, and store it in a (predefined) ndarray
        """
        if self.status:
            print(time.ctime(), ' - ', inspect.stack()[0][3])
            print('\t Reading through {} rows'.format(str(len(ref_data.index))))

        if input_file is None:
            input_file = self.input_file

        input_file_no_ext, _ = os.path.splitext(input_file)
        input_file = input_file_no_ext + '.data'

        kspace_shape = [len(x) for x in ref_data.index.levels]
        kspace_min_index = [min(list(x)) for x in ref_data.index.levels]

        max_bytes_kx = ref_data['size'].max() // 8

        # define matrix.
        temp_kspace = np.empty(kspace_shape + [max_bytes_kx], dtype=np.complex64)

        if self.debug:
            print('Shape of kspace is ', temp_kspace.shape)
            print('Kspace min index...', kspace_min_index)

        # loop over the data...
        with open(input_file, 'rb') as f:
            for i_index, i_row in ref_data.iterrows():
                i_index_translate = tuple(np.array(i_index) - kspace_min_index)
                temp_offset = int(i_row['offset'])
                byte_size = int(i_row['size'])

                # Read the actual data
                f.seek(temp_offset)
                temp_bin_file = np.fromfile(f, dtype='<f', count=byte_size // 4)
                # The real/imag value of the signal follow eachother
                temp_signal = temp_bin_file[0:][::2] + 1j * temp_bin_file[1:][::2]

                if self.debug:
                    print('Insert index ', i_index_translate)
                    print('Signal shape', temp_signal.shape)
                temp_kspace[i_index_translate] = temp_signal

        return temp_kspace

    def transform_scan_data(self, data_array):
        if self.status:
            print(time.ctime(), ' - ', inspect.stack()[0][3])

        # (Fourier) Transform the loaded data, if needed
        if self.complex:
            temp_data = data_array
        else:
            # Transform the data to image space
            temp_data = np.fft.ifft2(data_array)

        return temp_data

    def post_proc_image(self, image_data):
        # Here are some post processing steps... not always useful
        # Crop the oversampled parts away
        image_data = self.crop_image(image_data)
        # Rescale to reconstruction matrix size
        image_data = self.rescale_image(image_data)
        return image_data

    def crop_image(self, image_data):
        if self.status:
            print(time.ctime(), ' - ', inspect.stack()[0][3])
        # Correct for the cropped image shape.
        new_image_shape, old_image_shape = self.get_image_shape(self.parameter_table)

        # Test if we even need to crop something
        if not all(new_image_shape == old_image_shape):

            if self.parameter_table.get('kx_oversample_factor', default=0) > 1:
                n_lower = int((old_image_shape[-1] - new_image_shape[-1]) / 2)
                n_upper = int((old_image_shape[-1] + new_image_shape[-1]) / 2)
                image_data = np.take(image_data, indices=range(n_lower, n_upper), axis=-1)

            if self.parameter_table.get('ky_oversample_factor', default=0) > 1:
                n_lower = int((old_image_shape[-2] - new_image_shape[-2]) / 2)
                n_upper = int((old_image_shape[-2] + new_image_shape[-2]) / 2)
                ind_ky = self.index_col.index('ky')
                image_data = np.take(image_data, indices=range(n_lower, n_upper), axis=ind_ky)

            if self.parameter_table.get('kz_oversample_factor', default=0) > 1:
                n_lower = int((old_image_shape[-3] - new_image_shape[-3]) / 2)
                n_upper = int((old_image_shape[-3] + new_image_shape[-3]) / 2)
                ind_kz = self.index_col.index('kz')
                image_data = np.take(image_data, indices=range(n_lower, n_upper), axis=ind_kz)

        return image_data

    def rescale_image(self, image_data):
        """
        Is dit echt eerlijk om dit zo te doen?
        :param image_data:
        :return:
        """
        if self.status:
            print(time.ctime(), ' - ', inspect.stack()[0][3])

        rescaled_x = int(self.parameter_table.get('X-resolution', 1))
        rescaled_y = int(self.parameter_table.get('Y-resolution', 1))
        rescaled_z = int(self.parameter_table.get('Z-resolution', 1))
        rescaled_shape = [rescaled_z, rescaled_y, rescaled_x][-self.n_enc:]

        image_data_shape = np.array(image_data.shape)
        image_data_shape[-self.n_enc:] = rescaled_shape
        squeezed_image_data_shape = [x for x in image_data_shape if x != 1]
        new_axis_pos = np.where(image_data_shape == 1)[0]

        if self.status:
            print('Change size ', np.squeeze(image_data).shape, ' - to - ', squeezed_image_data_shape)

        # Squeezing it makes it so much faster
        if self.complex:
            temp_re = sktransf.resize(np.squeeze(np.real(image_data)), squeezed_image_data_shape)
            temp_cp = sktransf.resize(np.squeeze(np.imag(image_data)), squeezed_image_data_shape)
            squeezed_reshape = temp_re + 1j * temp_cp
        else:
            # Isnt this just doing the real part..?

            squeezed_reshape = sktransf.resize(np.squeeze(image_data), squeezed_image_data_shape)
        # But we do need to add back the axes to comply with self.index_col
        for i_ax in new_axis_pos:
            squeezed_reshape = np.expand_dims(squeezed_reshape, i_ax)

        return squeezed_reshape

    def get_roemer_image(self, image_data):
        if self.status:
            print(time.ctime(), ' - ', inspect.stack()[0][3])

        ind_chan = self.index_col.index('chan')
        n_chan = image_data.shape[ind_chan]
        # Because we are using a range here, the number of dimensions are preserved in np.take
        sel_chan = range(n_chan-8, n_chan)
        temp_img = np.take(image_data, sel_chan, axis=ind_chan)
        roemer_image = np.sqrt(np.sum(np.abs(temp_img) ** 2, axis=ind_chan, keepdims=True))
        return roemer_image

    def get_image_shape(self, param_data):
        """
        Returns the non oversampled image shape
        """
        if self.status:
            print(time.ctime(), ' - ', inspect.stack()[0][3])

        n_enc_dim = int(param_data['number_of_encoding_dimensions'])

        kx_factor = param_data.get('kx_oversample_factor', None)
        ky_factor = param_data.get('ky_oversample_factor', None)
        if n_enc_dim > 2:
            kz_factor = param_data.get('kz_oversample_factor', None)
            kz = int(np.diff(param_data.get('kz_range')) + 1)
        else:
            kz_factor = 1
            kz = 1

        kx = int(np.diff(param_data.get('kx_range', [1, 0]))+1)
        ky = int(np.diff(param_data.get('ky_range', [1, 0])) + 1)

        kspace_shape = np.array([kz, ky, kx])
        kspace_shape = kspace_shape[-n_enc_dim:]

        oversample_size = np.array([kz_factor, ky_factor, kx_factor])
        oversample_size = oversample_size[-n_enc_dim:]

        if any([x is None for x in oversample_size]):
            return kspace_shape, kspace_shape
        else:
            cor_shape = (kspace_shape/oversample_size).astype(int)

            return cor_shape, kspace_shape

    @staticmethod
    def _check_difference(parameter_table):
        # Used to check what is varying across mix/echo/loca
        param_list = []
        index_list = []
        for i, row in parameter_table.iterrows():
            param_list.append(dict(row))
            index_list.append(i)

        for k, _ in param_list[0].items():
            temp_list = [idict[k] for idict in param_list]
            if not all([temp_list[0] == x for x in temp_list]):
                mix, echo, loca = map(list, zip(*index_list))
                print(k)
                print(tabulate.tabulate([['mix'] + mix, ['echo'] + echo, ['loca'] + loca, ['value'] + temp_list]))

    def add_card_column(self, ref_data, param_data, n_card=None):
        """
        Replaces cardiac column in the provided DataFrame -- not sure if this is still useful
        """
        if n_card is None:
            n_card = int(param_data['number_of_cardiac_phases'].values)

        ref_data.loc[:, 'rperc'] = ref_data.loc[:, 'rtop']/(ref_data.loc[:, 'rr'] + 1e-06)
        # This is specific for cardiac phases...
        # I think I need to differentiate between different locations.. might also be per dynamic scan..??
        # ref_data.loc[:, 'card'] = ref_data['rperc'].apply(pd.cut, bins=n_card, labels=range(n_card))
        ref_data.loc[:, 'card'] = pd.cut(ref_data['rperc'], bins=n_card, labels=range(n_card))

        return ref_data


if __name__ == "__main__":
    import helper.plot_fun as hplotf
    import helper.plot_class as hplotc

    # dir_data_list = '/media/bugger/MyBook/data/7T_scan/cardiac/20180926_scan'
    # dir_data_list = '/media/bugger/MyBook/data/7T_scan/cardiac/20181212_scan'
    dir_data_list = '/media/bugger/MyBook/data/7T_scan/cardiac/20190227_scan'
    list_files = [x for x in os.listdir(dir_data_list) if x.endswith('list')]
    for sel_file in list_files[0:2]:
        path_data = os.path.join(dir_data_list, sel_file)
        DL_obj = DataListImage(path_data, debug=False)
        data_array, img_array = DL_obj.get_image_data()
        hplotc.SlidingPlot(data_array)
        hplotc.ListPlot(np.squeeze(img_array), title=sel_file, augm='np.abs')
        noise_data = DL_obj.get_std_data('NOI')
        noise_data.shape

