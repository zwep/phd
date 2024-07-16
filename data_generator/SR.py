import scipy.ndimage
import scipy.signal
import os
import PIL
import numpy as np
import scipy.ndimage
from PIL import Image
from skimage import transform as sktransform
from data_generator.Generic import DatasetGenericComplex, DatasetGeneric
from helper import array_transf as harray
import torch
import scipy.signal


class DataGeneratorSR(DatasetGeneric):
    """
    A very simple dataset.. where we have a noisy image and a clear output
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
        'transform_type' - keep the 'complex' values, or transform to 'real' or others..
        'transform_type_target' - similar to `transform_type` but for target value only
        'stack_ax' - axis on which the complex parts are being stacked
        'concatenate_complex' - if True transform 8 complex coils to 16 channels. If false, transforms to (2, 8) channels
        """
        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, **kwargs)

        n_kernel = kwargs.get('n_kernel', 5)
        self.n_scale = kwargs.get('n_scale', 3)
        self.smooth_kernel = np.ones((n_kernel, n_kernel)) / n_kernel ** 2

    def __getitem__(self, index):
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']
        mask_dir = self.container_file_info[sel_dataset]['mask_dir']

        # Hopefully this is a fool-proof way of reading images..
        # Some images had bad magic numbers or byte numbers or so...
        read_image_ind = False
        offset_counter = 0
        while read_image_ind is False:
            try:
                i_file = file_list[index + offset_counter]
                clear_image_file = os.path.join(input_dir, i_file)
                pillow_obj = Image.open(clear_image_file)
                read_image_ind = True
            except PIL.UnidentifiedImageError:
                print('Reading image failed: ', i_file)
                offset_counter += 1

        high_res_image_array = np.array(pillow_obj.convert('LA'))
        high_res_image_array = np.take(high_res_image_array, 0, axis=-1)
        high_res_image_array = sktransform.resize(high_res_image_array, self.img_input_shape[-2:], order=3)
        high_res_image_array = harray.scale_minmax(high_res_image_array)


        image_smooth = scipy.signal.convolve2d(high_res_image_array, self.smooth_kernel, mode='same')
        low_res_image_array = scipy.ndimage.zoom(image_smooth, 1 / self.n_scale, order=3)

        low_res_image_tensor = torch.from_numpy(low_res_image_array).float()
        high_res_image_tensor = torch.from_numpy(high_res_image_array).float()
        container = {'input': low_res_image_tensor[np.newaxis], 'target': high_res_image_tensor[np.newaxis]}

        return container


if __name__ == "__main__":

    data_gen = DataGeneratorSR(ddata='/home/bugger/Documents/data/celeba', input_shape=(256, 256), debug=True,
                               file_ext='jpg', n_kernel=5, n_scale=4)

    container = data_gen.__getitem__(0)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(container['input'])
    plt.figure()
    plt.imshow(container['target'][0])
