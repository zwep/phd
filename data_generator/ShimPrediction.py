import os
import numpy as np
import torch.utils.data
import data_generator.Generic as data_gen
import helper.array_transf as harray
import helper.misc as hmisc
"""

"""




class DataGeneratorShimPrediction(data_gen.DatasetGenericComplex):
    """
    Predict zeh shim....
    """

    def __init__(self, ddata, input_shape=None, target_shape=None,
                 shuffle=True, dataset_type='train', file_ext='npy', transform=None, **kwargs):
        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, transform=transform, **kwargs)

        # If True, then randomly decides if we transform the data to:
        # (128, 128), (256, 256), (512, 512), (1024, 1024)
        self.transform_resize = kwargs.get('transform_resize', False)
        self.resize_list = [(50, 50)]
        self.resize_index = 0

    def __getitem__(self, index):
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        # target_dir = self.container_file_info[sel_dataset]['target_dir']
        mask_dir = self.container_file_info[sel_dataset]['mask_dir']
        index = index % len(file_list)
        i_file = file_list[index]

        input_file = os.path.join(input_dir, i_file)
        temp_array = hmisc.load_array(input_file)
        ny, nx = temp_array.shape[-2:]
        new_shape = (-1, ny, nx)
        input_real = temp_array[0].reshape(new_shape)
        input_imag = temp_array[1].reshape(new_shape)
        input_array = np.concatenate([input_real, input_imag])

        # Lets create a very simple mask for now..
        # mask_file = os.path.join(mask_dir, i_file)
        # if os.path.isfile(mask_file):
        #     mask_array = hmisc.load_array(mask_file)
        # else:
        mask_array = np.ones((1, ny, nx))
        # mask_array[ny//2-10:ny//2+10, nx//2-10:nx//2+10] = 1

        if self.debug:
            print('Shape of A_cpx', input_array.shape)
            print('Shape of input_real', input_real.shape)
            print('Shape of input_imag', input_imag.shape)
            print('Resemblence imag', np.sum(input_imag[0] == input_array[64]))
            print('Resemblence real', np.sum(input_real[0] == input_array[0]))

        # Normalize after checking if we took the right order of things....
        input_array = harray.scale_minmax(input_array)
        # input_array = input_array / np.max(input_array)

        target_array = np.abs(temp_array).sum(axis=0).sum(axis=0)
        target_array = harray.scale_minmax(target_array)

        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(target_array).float()
        mask_tensor = torch.from_numpy(mask_array).float()
        container = {'input': input_tensor, 'target': target_tensor, 'mask': mask_tensor}

        return container


if __name__ == "__main__":
    import helper.plot_class as hplotc
    # Test this thing..
    ddata = '/media/bugger/MyBook/data/dataset/b1_shim_series'
    data_gen_obj = DataGeneratorShimPrediction(ddata=ddata, dataset_type='train', file_ext='h5')
    cont = data_gen_obj.__getitem__(0)
    cont['input'].shape
    cont['target'].shape
    cont['mask'].shape
    # hplotc.ListPlot([cont['mask']])
    hplotc.SlidingPlot(cont['input'])
    hplotc.SlidingPlot(np.array(cont['input'])[:64] + 1j * np.array(cont['input'])[64:])
    # hplotc.SlidingPlot((np.array(cont['input'])[:64] + 1j * np.array(cont['input'])[64:]).sum(axis=0))
    # hplotc.ListPlot([np.abs((np.array(cont['input'])[:64] + 1j * np.array(cont['input'])[64:])).sum(axis=0), cont['target']])
