
import os
import numpy as np
import pandas as pd
import warnings
import re

"""
Files and functions to read raw data from the MR scanner
"""


class ReadCpx:
    # These lines are used to check if we are dealing with the .par-file of a .rec-file or not...
    img_inf_def_line = '# === IMAGE INFORMATION DEFINITION =============================================\n'
    img_inf_line = '# === IMAGE INFORMATION ==========================================================\n'
    def __init__(self, input_file):
        """

        :param input_file: SHould specify the whole path.. can be with or without the extension
        """
        self.input_file = input_file
        try:
            self.header = self.get_cpx_header(input_file)
            list_slices = [int(x) for x in self.header['loc'].value_counts().index.tolist()]
            self.loc_list = sorted(list_slices)
        except IndexError:
            print('Getting header failed')
            self.header = None
            self.loc_list = None
        except FileNotFoundError:
            warnings.warn(f'The .cpx file is not found {input_file}')
            self.header = None
            self.loc_list = None

        # This is a sub selection of the columns we want to return..
        self.sub_index_col = ['chan', 'loc', 'slice', 'hps', 'echo', 'dyn', 'mix']

    def get_par_file(self, input_file=None):
        if input_file is None:
            input_file = self.input_file

        input_file, ext = os.path.splitext(input_file)
        with open(input_file + '.par', 'r') as f:
            file_lines = f.readlines()

        general_lines = [x[1:].strip().split(":") for x in file_lines if x.startswith('.')]
        general_lines_dict = {x[0].strip(): x[1].strip() for x in general_lines}

        if self.img_inf_line in file_lines:
            image_information_dict = self.get_par_file_rec(input_file)
            general_lines_dict.update(image_information_dict)

        return general_lines_dict

    def get_par_file_rec(self, input_file):
        # The par-file associated with a .rec-file has more information that needs to be parsed..
        image_information_dict = {}
        with open(input_file + '.par', 'r') as f:
            file_lines = f.readlines()

            index_img_inf_def = file_lines.index(self.img_inf_def_line)
            index_img_inf = file_lines.index(self.img_inf_line)

            img_inf_def_list = file_lines[index_img_inf_def + 3:index_img_inf - 1]
            img_inf_list = file_lines[index_img_inf + 3].split()
            start_index = 0
            end_index = 0
            for x in img_inf_def_list:
                # Remove # and potential (imagekey!) characters. No idea what (imagekey!) really means.
                x = re.sub('\#|\(imagekey!\)', '', x).strip()
                re_result = re.findall('\(([0-9])\*\w+\)', x)
                if re_result:
                    number_of_items = int(re_result[0])
                else:
                    number_of_items = 1
                x = re.sub('\s\s+\(.*\w+\)', '', x).strip()
                end_index += number_of_items
                temp_dict = {x: img_inf_list[start_index: end_index]}
                image_information_dict.update(temp_dict)
                start_index += number_of_items
                end_index = start_index

        return image_information_dict

    def get_cpx_header(self, input_file=None):
        """
        Defines the header.. and the offset!
        Does not need the .par-file...
        """
        if input_file is None:
            input_file = self.input_file

        input_file, ext = os.path.splitext(input_file)
        with open(input_file + '.cpx') as f_id:
            h1 = np.fromfile(f_id, count=15, dtype=np.int32)
            factor = np.fromfile(f_id, count=2, dtype=np.float32)  # Needs to be read of
            h2 = np.fromfile(f_id, count=111, dtype=np.int32)

        # Quanitities needed to determine the header offset..
        res_x = h1[10]
        res_y = h1[11]
        compression = h1[13]
        matrix_data_blocks = h1[12]
        offset = h2[25] or h1[9]  # Offset is from either two header 'files'

        # Count the amount of images we have in the file..
        image_exist = 1
        counter_img = 0
        with open(input_file + '.cpx') as f_id:
            while image_exist:
                # header_offset = (res_x * res_y * 8 /compression + offset)*i
                header_offset = (matrix_data_blocks * 512 + offset) * counter_img
                f_id.seek(header_offset)
                h1 = np.fromfile(f_id, count=15, dtype=np.int32)
                image_exist = h1[8]
                counter_img = counter_img + 1
        num_img = counter_img - 1

        # Collect all the header information
        # We have identified most of the positions..
        h1_index = ['mix', 'loc', 'slice', 'prepdir', 'echo', 'hps', 'dyn', 'segm', 'img_ind', 'offset',
                    'resx', 'resy', 'compression', 'seqnum', 'unknown']
        h2_index = ['segm2', 'chan', 'syncnum', 'flip']
        factor_index = ['factor', 'factor2']
        index_col = h1_index + h2_index + factor_index
        n_chan = len(index_col)

        # Define empty header
        header = np.zeros((num_img, n_chan))
        with open(input_file + '.cpx') as f_id:
            for i in range(num_img):
                header_offset = (matrix_data_blocks * 512 + offset) * i
                f_id.seek(header_offset)
                h1 = np.fromfile(f_id, count=15, dtype=np.int32).tolist()
                factor = np.fromfile(f_id, count=2, dtype=np.float32).tolist()
                h2 = np.fromfile(f_id, count=111, dtype=np.int32).tolist()

                header[i, :] = h1[:len(h1_index)] + h2[:len(h2_index)] + factor[:len(factor_index)]
                header[i, 9] = h2[25] or h1[9]  # Data offset
                if h1[8] == 0:
                    'Header Problem!! Too many images calculated'
                    break

            # Check the last image as well
            last_header_offset = (matrix_data_blocks * 512 + offset) * num_img
            f_id.seek(last_header_offset)
            h1 = np.fromfile(f_id, count=15, dtype=np.int32)
            if h1[8] != 0:
                print('Header Problem')

        # Return the info as a DataFrame
        header_pd = pd.DataFrame(header)
        header_pd.columns = index_col
        return header_pd

    def get_cpx_img(self, sel_loc=None):
        input_file, ext = os.path.splitext(self.input_file)
        n_img = len(self.loc_list)
        img_range = sel_loc or range(n_img)
        self.header = self.header[self.header['loc'].isin(img_range)]

        nx, ny = int(self.header['resx'].unique()), int(self.header['resy'].unique())
        return_size = [len(x.unique()) for y, x in self.header[self.sub_index_col].items()] + [ny, nx]
        img_array = np.empty(return_size, dtype=np.complex)

        with open(input_file + '.cpx') as f_id:
            for i, irow in self.header.iterrows():
                # Get the right offset..
                offset = irow['offset']
                # Get the right index...
                # I hoped this could've been done better or easier... like with logical indexing or so..
                ind_chan, ind_loc, ind_slice, ind_hps, ind_echo, ind_dyn, ind_mix = irow[self.sub_index_col].astype(int).tolist()

                f_id.seek(offset-512)
                h1 = np.fromfile(f_id, count=15, dtype=np.int32)
                factor = np.fromfile(f_id, count=2, dtype=np.float32)
                h2 = np.fromfile(f_id, count=111, dtype=np.int32)

                res_x = h1[10]  # Resolution x
                res_y = h1[11]  # Resolution y
                compression = h1[13]  # Compression

                f_id.seek(offset)
                if compression == 1:
                        img_data = np.fromfile(f_id, count=res_x * res_y * 2, dtype=np.float32)
                elif compression == 2:
                        img_data = np.fromfile(f_id, count=res_x * res_y * 2, dtype=np.int16)
                        img_data = factor[1] + factor[0] * img_data
                elif compression == 4:
                        img_data = np.fromfile(f_id, count=res_x * res_y * 2, dtype=np.int8)
                        img_data = factor[1] + factor[0] * img_data
                else:
                    img_data = []

                img_data = img_data[::2] + 1j * img_data[1::2]
                img_data = img_data.reshape((res_y, res_x))
                img_array[ind_chan, self.loc_list.index(ind_loc), ind_slice, ind_hps, ind_echo, ind_dyn, ind_mix] = img_data

        return img_array
    # Alias
    read_cpx_img = get_cpx_img


def apply_shim_4d(x, cpx_shim=None, n_chan=8, axis=0):
    # cpx shim is the complex valued shim values
    if cpx_shim is None:
        amp = np.ones(n_chan)
        phase = np.random.normal(0, 0.5 * np.sqrt(np.pi), size=n_chan)
        cpx_shim = np.array(
            [r * np.exp(1j * (phi + np.random.normal(0, 0.02 * np.sqrt(np.pi)))) for r, phi in zip(amp, phase)])

    if axis == 0:
        x = np.einsum("tsmn, t -> smn", x, cpx_shim)
    else:
        x = np.einsum("tsmn, s -> tmn", x, cpx_shim)

    return x


if __name__ == "__main__":
    import helper.plot_class as hplotc
    import shimming.b1shimming_single as b1shim_tool
    cpx_dir = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_06/V9_16935'
    cpx_file_list = [os.path.join(cpx_dir, x) for x in os.listdir(cpx_dir) if x.endswith('cpx')]

    # Survey 1
    cpx_file_survey = cpx_file_list[0]
    cpx_survey_obj = ReadCpx(cpx_file_survey)
    img_survey = cpx_survey_obj.get_cpx_img()
    hplotc.SlidingPlot(img_survey.sum(axis=0))

    # Survey 2
    cpx_file_survey_after = cpx_file_list[1]
    cpx_survey_after_obj = ReadCpx(cpx_file_survey_after)
    img_survey_after = cpx_survey_after_obj.get_cpx_img()
    hplotc.SlidingPlot(img_survey_after.sum(axis=0))

    cpx_shim_series = [x for x in cpx_file_list if 'shim' in x][0]
    cpx_shim_series_obj = ReadCpx(cpx_shim_series)
    cpx_shim_series_array = cpx_shim_series_obj.get_cpx_img()
    cpx_shim_series_array = np.rot90(cpx_shim_series_array, axes=(-2, -1))
    cpx_shim_array = np.squeeze(cpx_shim_series_array)

    no_shim_array = cpx_shim_array.sum(axis=0).sum(axis=0)

    mask_obj = hplotc.MaskCreator(np.squeeze(cpx_shim_series_array).sum(axis=0))
    mask_array = mask_obj.mask

    import helper.array_transf as harray

    init_axis_0 = None
    init_axis_1 = None

    for _ in range(4):
        first_axis = apply_shim_4d(cpx_shim_array, cpx_shim=init_axis_0, axis=0)
        shim_obj_0 = b1shim_tool.ShimmingProcedure(input_array=first_axis, mask=mask_array, str_objective='flip_angle')
        x_opt_0, final_value = shim_obj_0.find_optimum()
        b1_array_0 = harray.apply_shim(first_axis, cpx_shim=x_opt_0)

        second_axis = apply_shim_4d(cpx_shim_array, cpx_shim=init_axis_1, axis=1)
        shim_obj_1 = b1shim_tool.ShimmingProcedure(input_array=second_axis, mask=mask_array, str_objective='flip_angle')
        x_opt_1, _ = shim_obj_1.find_optimum()
        b1_array_1 = harray.apply_shim(second_axis, cpx_shim=x_opt_1)

        init_axis_0 = x_opt_1
        init_axis_1 = x_opt_0

        hplotc.ListPlot([[no_shim_array, b1_array_1 * (0.5 + mask_array)]], augm='np.abs', ax_off=True)