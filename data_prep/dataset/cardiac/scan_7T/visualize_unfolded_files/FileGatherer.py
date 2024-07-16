"""
Similar to another class..
with this thing we can gather all the files....
"""

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import helper.plot_class as hplotc
import helper.misc as hmisc
import os
import re
import h5py


class FileGather:
    def __init__(self, data_dir, file_ext='mat'):
        self.data_dir = data_dir
        self.file_ext = file_ext
        self.file_list_ext = self.get_file_list(data_dir=data_dir)
#
    def get_file_list(self, data_dir):
        """
        This get us our file list in a nice list.

        :param data_dir:
        :return:
        """
        file_list = []
        for d, _, f in os.walk(data_dir):
            filter_f = [x for x in f if x.endswith(self.file_ext)]
            if len(filter_f):
                for i_file in filter_f:
                    file_path = os.path.join(d, i_file)
                    file_list.append(file_path)
        return file_list
#
    def get_shape_list(self, data_key='data'):
        shape_list = []
        for i_file in self.file_list_ext:
            # This does not work when multiple keys exist in the directory...
            loaded_array = hmisc.load_array(i_file, data_key=data_key)
            temp_shape = loaded_array.shape
            shape_list.append(temp_shape)
        return shape_list

    """
    self.plot_dir = self.data_dir + '_png'
    
    I removed this part
    # Create destination dir
        d_png = re.sub(self.data_dir, self.plot_dir, d)
        if not os.path.isdir(d_png):
            os.makedirs(d_png)
    """


class PlotFileList:
    def __init__(self, file_list, source_dir, dest_dir):
        # Make sure we dont see it
        plt.ioff()
        import matplotlib
        matplotlib.use('Agg')
        self.file_list = file_list
        self.source_dir = source_dir
        self.dest_dir = dest_dir

    def plot_all_files(self):
        for i_file in self.file_list:
            self.plot_file(i_file)
            hplotc.close_all()

    def plot_file(self, sel_file):
        # Swap the currente extension with .png
        file_ext = hmisc.get_ext(sel_file)
        file_ext = re.sub('\.', '\.', file_ext)
        sel_file_png = re.sub(file_ext, '.png', sel_file)
        sel_file_png = re.sub(self.source_dir, self.dest_dir, sel_file_png)
        dest_dir = os.path.dirname(sel_file_png)
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        # Load the array
        loaded_array = self._get_array(sel_file)
        if loaded_array is None:
            return
        loaded_array = self.correct_dimensions(loaded_array)
        nx, ny, _, ncoil, _, ncard, _, nloc = loaded_array.shape
        loaded_array = np.abs(loaded_array).sum(axis=3, keepdims=True)
        # Reduce the size to a 2D array. I could possibly use the get_all_mid_slice function
        # But not sure which index to pick from there..
        plot_array = loaded_array[:, :, 0, 0, 0, ncard // 2, 0, nloc//2]
        fig_obj = hplotc.ListPlot(plot_array)
        fig_obj.figure.savefig(sel_file_png, bbox_inches='tight', pad_inches=0.0)

    def _get_array(self, i_file):
        # This is very specific for the mat-files I stored...
        loaded_array = hmisc.load_array(i_file, data_key='data_param')
        if isinstance(loaded_array, dict):
            loaded_array = hmisc.load_array(i_file, data_key='vartosave')

        if isinstance(loaded_array, dict):
            # Some are still not properly loaded...
            return None
        else:
            # Unpack a packed mat-object.
            if (loaded_array.ndim == 2) and (loaded_array.shape[0] == 5):
                loaded_array = loaded_array[0][0]
        return loaded_array

    @staticmethod
    def correct_dimensions(loaded_array):
        """
        Correct the dimensions...

        :param loaded_array
        :return:
        """
        # Check dimensions to distinguish files WITH individual coils, locations, etc.
        ndim = loaded_array.ndim
        if ndim == 8:
            pass
            # nx, ny, _, ncoil, _, ncard, _, nloc = loaded_array.shape
        elif ndim == 6:
            # Adding axes to make it similar to the 8 dimension case
            # Hoping that this makes processing easier.
            loaded_array = np.expand_dims(loaded_array, axis=(-2, -1))
        elif ndim == 5:
            # This dimension type uses dynamics instead of ncard
            # Therefore we add an axis BEFORE this dimension, so that it aligns with ncard.
            loaded_array = np.expand_dims(loaded_array, axis=(-4, -2, -1))
        elif ndim == 4:
            # Now we have (loc, card, nx, ny) data
            pass
        elif ndim == 3:
            # We dont have a location now..
            loaded_array = loaded_array[None]
        elif ndim == 2:
            loaded_array = np.expand_dims(loaded_array, axis=(2, 3, 4, 5, 6, 7))
        else:
            print("Unknown number of dimensions ", loaded_array.shape)
            return None
        return loaded_array
