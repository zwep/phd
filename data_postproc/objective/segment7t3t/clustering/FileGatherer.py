"""
Here we define a class that gathers the necessary files
"""

import skimage
import itertools
import helper.plot_class as hplotc
import collections
import os
import helper.misc as hmisc
import re
import numpy as np
import skimage.transform as sktransform
import helper.array_transf as harray
from skimage.util import img_as_ubyte

from skimage.metrics import structural_similarity
import scipy.stats
import helper.metric as hmetric
from objective_configuration.segment7T3T import DATASET_LIST, DATASET_SYNTH_LIST
from objective_configuration.segment7T3T import DICT_DATASET_DLABEL, DICT_DATASET_DIMG, DLOG
from loguru import logger

logger.add(os.path.join(DLOG, 'FileGatherer.log'))


class FileGatherer:
    resize_shape = (256, 256)

    def __init__(self, dict_dataset, debug=False, estimate_shape=False):
        self.estimate_shape = estimate_shape
        self.dataset_list = list(dict_dataset.keys())
        self.dataset_dir = dict_dataset
        self.debug = debug
        self.file_dict_list = self.get_files()

    def get_unique_files(self, sel_dataset, list_files):
        if 'mm1' in sel_dataset:
            re_obj = re.compile('(?:^|_)([A-Z].*)_loc_')
        elif 'acdc' in sel_dataset:
            re_obj = re.compile('patient([0-9]{3})_')
        elif 'mm2' in sel_dataset:
            re_obj = re.compile('^([0-9]{3})_')
        elif '7t' in sel_dataset:
            re_obj = re.compile('_subject_([0-9]{4})_')
        elif 'kaggle' in sel_dataset:
            re_obj = re.compile('^([0-9]+)-')
        else:
            print('Unknown dataset name', sel_dataset)
            return None

        unique_patient_files = []
        # Waarom hier zo veel haakjes..?
        patient_id_list = list(set([re_obj.findall(x)[-1] for x in list_files]))
        for i_patient in patient_id_list:
            i_patient_files = sorted([x for x in list_files if i_patient in x])
            n_files = len(i_patient_files)
            any_loc_info = any(['loc' in x for x in i_patient_files])
            if any_loc_info:
                # print('Found loc info: ')
                regex_result = [re.findall('_loc_([0-9]+)', x) for x in i_patient_files]
                # Cast it....
                regex_result = [int(x[0]) for x in regex_result]
                # print('Location ind ', regex_result)
                mid_index = np.argsort(regex_result)[n_files // 2]
                selected_file = i_patient_files[mid_index]
            else:
                selected_file = i_patient_files[n_files//2]

            unique_patient_files.append(selected_file)
        return unique_patient_files

    def get_files(self):
        file_dict_list = {}
        for i_dataset, i_dir in self.dataset_dir.items():
            logger.info(f'\nDirectory {i_dir}')
            list_files = os.listdir(i_dir)
            list_files = sorted(list_files)
            list_files = self.get_unique_files(i_dataset, list_files)
            logger.info(f'\tNumber of files {len(list_files)}')
            if self.estimate_shape:
                list_of_shapes = []
                for i_file in list_files:
                    file_path = os.path.join(i_dir, i_file)
                    file_shape = hmisc.load_array(file_path).shape
                    list_of_shapes.append(file_shape)
                #
                ndim_list = [len(x) for x in list_of_shapes]
                logger.info('\tList of shapes')
                hmisc.print_dict(collections.Counter(list_of_shapes))
                logger.info('\tNdim list')
                hmisc.print_dict(collections.Counter(ndim_list))
            logger.info('\tSample files')
            logger.info(', '.join(list_files[:10]))
            file_dict_list[i_dataset] = [os.path.join(i_dir, x) for x in list_files]
        return file_dict_list

    def load_array(self, sel_file):
        if os.path.isfile(sel_file):
            file_array = np.squeeze(hmisc.load_array(sel_file))
            if file_array.ndim == 3:
                n_slice = file_array.shape[-1]
                sel_slice = n_slice // 2
                sel_file_array = file_array[:, :, sel_slice]
            else:
                sel_file_array = file_array
            # Convert to 8-bits image
            sel_file_array = img_as_ubyte(harray.scale_minmax(sel_file_array))
            sel_file_array = sktransform.resize(sel_file_array, output_shape=self.resize_shape,
                                                anti_aliasing=False, preserve_range=True)
            return sel_file_array

    def load_n_array(self, dataset=None, n=None):
        if dataset is None:
            print('Choose one of the following ', self.file_dict_list.keys())
        else:
            dataset = dataset.lower()

        if n is None:
            n = len(self.file_dict_list[dataset])
        else:
            n_max = len(self.file_dict_list[dataset])
            n = min(n, n_max)
        array_list = []
        for ii in range(n):
            sel_file = self.file_dict_list[dataset][ii]
            logger.debug(f'Loading {sel_file}')
            sel_file_array = self.load_array(sel_file=sel_file)
            array_list.append(sel_file_array)
        data_array = np.array(array_list)
        return data_array


if __name__ == "__main__":
    filegather_object = FileGatherer(DICT_DATASET_DIMG, debug=True)
    list_of_arrays = filegather_object.load_n_array(dataset='7T', n=6)
    fig_obj = hplotc.SlidingPlot(list_of_arrays)
