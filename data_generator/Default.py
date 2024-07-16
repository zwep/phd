import os
import torchio
import PIL
import numpy as np
import scipy.ndimage
from PIL import Image
from skimage import transform as sktransform

from data_generator.Generic import DatasetGenericComplex, DatasetGeneric
from helper import array_transf as harray
import torch
import scipy.signal


class DataGeneratorNoise(DatasetGenericComplex):
    """
    A very simple dataset.. where we have a noisy image and a clear output
    """
    def __init__(self, ddata, input_shape, target_shape=None, shuffle=True, dataset_type='train', file_ext='npy', **kwargs):
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

        self.n_rep = kwargs.get('n_rep', 20)
        self.noise_mode = kwargs.get('noise_mode', 'additive')
        self.torchio_mode = kwargs.get('torchio_mode', None)

    def __getitem__(self, index):
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']
        mask_dir = self.container_file_info[sel_dataset]['mask_dir']

        # Hopefully this is a fool-proof way of reading images..
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

        clear_image_array = np.array(pillow_obj.convert('LA'))
        clear_image_array = np.take(clear_image_array, 0, axis=-1)
        clear_image_array = sktransform.resize(clear_image_array, self.img_input_shape[-2:])
        clear_image_array = harray.scale_minmax(clear_image_array)

        noise_mask = np.array([np.random.standard_normal(size=self.img_input_shape) for _ in range(self.n_rep)]).std(axis=0)
        if self.noise_mode == 'additive':
            noisy_image_array = clear_image_array + noise_mask
        elif self.noise_mode == 'multiplicative':
            noisy_image_array = clear_image_array * noise_mask
        else:
            noisy_image_array = None

        noisy_image_array = harray.scale_minmax(noisy_image_array)
        noisy_image_tensor = torch.from_numpy(noisy_image_array).float()
        clear_image_tensor = torch.from_numpy(clear_image_array).float()
        container = {'input': noisy_image_tensor[np.newaxis], 'target': clear_image_tensor[np.newaxis]}
        ## Add Torch.io transfomers
        # Easy transforms
        if self.torchio_mode is None:
            return container
        elif self.torchio_mode == 'easy':
            torchio_transform = torchio.Compose([torchio.RandomFlip(), torchio.RandomBlur(), torchio.RandomNoise()])
            container['input'] = torchio_transform(noisy_image_tensor[np.newaxis, :, :, np.newaxis])[:, :, :, 0]
            return container
        elif self.torchio_mode == 'hard':
            torchio_transform = torchio.Compose([torchio.RandomBiasField(), torchio.RandomElasticDeformation()])
            container['input'] = torchio_transform(noisy_image_tensor[np.newaxis, :, :, np.newaxis])[:, :, :, 0]
            return container

        return container


if __name__ == "__main__":
    data_gen = DataGeneratorNoise(ddata='/media/bugger/MyBook/data/celeba', input_shape=(64, 64), debug=True,
                                  file_ext='jpg', n_rep=100, noise_mode='additive')

    container = data_gen.__getitem__(1)
    import helper.plot_class as hplotc
    hplotc.ListPlot([container['input'][0], container['target'][0]])

    import helper.misc as hmisc
    target_dir = "/local_scratch/sharreve/plain_data/cats_dogs"
    hmisc.create_datagen_dir(target_dir, type_list=('test', 'validation', 'train'), data_list=('input', 'target'))