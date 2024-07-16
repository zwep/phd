# encoding: utf-8

import helper.array_transf as harray
import helper.plot_fun as hplotf
import helper.plot_class as hplotc

import helper_torch.transforms as htransform
import os
import numpy as np
import scipy.stats
import importlib
import torch.utils.data
import re
import data_generator.Generic as data_gen
import scipy.signal
import h5py

"""
Here we have the data generators for the Survey 2 B1 maps

either single channel.. or all channels..
"""


class DataGeneratorBmin2Bplus(data_gen.DatasetGenericComplex):
    def __init__(self, ddata, input_shape=None, target_shape=None,
                 shuffle=True, dataset_type='train', file_ext='npy', transform=None, **kwargs):
        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, transform=transform, **kwargs)

        self.transform_resize = kwargs.get('transform_resize', False)
        self.multiply_rho = kwargs.get('multiply_rho', False)
        self.resize_list = [(256, 256)]
        self.resize_index = len(self.resize_list) - 1

    def __getitem__(self, index):
        """Generate one batch of data"""
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']
        target_clean_dir = self.container_file_info[sel_dataset]['target_dir'] + "_clean"
        mask_dir = self.container_file_info[sel_dataset]['mask_dir']
        i_file = file_list[index]

        input_file = os.path.join(input_dir, i_file)
        target_file = os.path.join(target_dir, i_file)
        mask_file = os.path.join(mask_dir, i_file)
        target_clean = os.path.join(target_clean_dir, i_file)

        with h5py.File(input_file, 'r') as h5_obj:
            max_slice = h5_obj['data'].shape[0]

        sel_slice = np.random.randint(max_slice)

        with h5py.File(input_file, 'r') as f:
            input_array = np.array(f['data'][sel_slice])

        with h5py.File(mask_file, 'r') as f:
            mask_array = np.array(f['data'][sel_slice])

        with h5py.File(target_file, 'r') as f:
            target_array = np.array(f['data'][sel_slice])

        with h5py.File(target_clean, 'r') as f:
            rho_array = np.array(f['data'][sel_slice])

        rho_array = harray.scale_minmax(rho_array)
        # Convert input to CPX
        input_array = input_array[0] + 1j * input_array[1]
        if self.multiply_rho:
            input_array = input_array * mask_array * rho_array[None]
        else:
            input_array = input_array * mask_array
        # Convert target to CPX
        target_array = target_array[0] + 1j * target_array[1]
        target_array = target_array * mask_array

        # Change coil order of both input and target
        coil_order = np.arange(8)
        np.random.shuffle(coil_order)
        input_array = input_array[coil_order]
        target_array = target_array[coil_order]

        # Normalize input
        input_array = harray.scale_minmax(input_array, is_complex=True)
        # Transform complex input to stacked version
        input_array = self.transform_complex(input_array, stack_ax=self.stack_ax)

        # Normalize target
        target_array = harray.scale_minmax(target_array, is_complex=True)




        if self.transform_type_target is None:
            target_array = self.transform_complex(target_array, stack_ax=self.stack_ax)
        else:
            target_array = self.transform_complex(target_array,
                                                  transform_type=self.transform_type_target,
                                                  stack_ax=self.stack_ax)
        if self.transform_resize:
            # This can probably be done prettier.. but yeah..
            resize_shape = self.resize_list[self.resize_index]
            input_array = self.perform_resize(input_array, resize_shape=resize_shape)
            target_array = self.perform_resize(target_array, resize_shape=resize_shape)
            mask_array = self.perform_resize(mask_array, resize_shape=resize_shape)

        n_c = target_array.shape[0]
        mask_array = np.tile(mask_array, (n_c, 1, 1))
        input_tensor = torch.as_tensor(input_array).float()
        target_tensor = torch.as_tensor(target_array).float()
        mask_tensor = torch.as_tensor(mask_array).float()
        container_dict = {'input': input_tensor, 'target': target_tensor, 'mask': mask_tensor}

        random_seed_fixed = np.random.randint(123456789)
        if self.transform_compose is not None:  # and self.dataset_type == 'train':
            for key, value in container_dict.items():
                torch.manual_seed(random_seed_fixed)
                temp = value
                # The TorchIO transform stuff needs 3D arrays.. and we want 2D stuff
                # Therefore we add and remove an axis
                # It expects input (channels, x, y, z)
                # I convert all my data to (channels, x, y) already.. so this is fine
                temp = self.transform_compose(temp[..., None])[..., 0]
                # for i_transform in self.transform_compose.transforms:
                #     # Only perform Random Erasing on the input
                #     # (Or: when we have something unequal to input, continue)
                #     if i_transform._get_name() == 'RandomErasing' and key != 'input':
                #         continue
                #
                #     temp = i_transform(temp)
                container_dict[key] = temp
                
        return container_dict 

class DataSetSurvey2B1_flavio(data_gen.DatasetGenericComplex):
    def __init__(self, ddata, input_shape, target_shape=None,
                 shuffle=True, dataset_type='train', file_ext='npy', **kwargs):

        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, **kwargs)

        self.masked = kwargs.get('masked', False)
        self.random_phase = kwargs.get('random_phase', False)

    def __getitem__(self, index):
        """Generate one batch of data"""

        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']
        mask_dir = self.container_file_info[sel_dataset]['mask_dir']
        i_file = file_list[index]

        input_file = os.path.join(input_dir, i_file)
        target_file = os.path.join(target_dir, i_file)
        mask_dir = re.sub('input', 'mask', input_dir)
        mask_file = os.path.join(mask_dir, i_file)

        # Will be of shape 8, X, Y
        input_array = np.load(input_file)
        target_array = np.load(target_file)
        mask_array = np.load(mask_file)[np.newaxis]
        # Only correct `isclose` for values OUTSIDE the mask.
        mask_int = ((1 - mask_array) == 1).astype(int)

        if self.random_phase:
            random_phase_input = np.random.uniform(-1, 1, size=8) + 1j * np.random.uniform(-1, 1, size=8)
            input_array = input_array * random_phase_input[:, None, None]

            input_array_close = np.isclose(input_array, 0).astype(int)
            input_array_outside = (input_array_close * mask_int).astype(bool)
            input_array[input_array_outside] = 0

        scale_input = np.max(np.abs(input_array))
        input_array = input_array / scale_input
        input_array = self.transform_complex(input_array, stack_ax=self.stack_ax)

        if self.input_is_output:
            target_array = np.copy(input_array)
        else:
            target_array = target_array / scale_input
            # target_array = target_array / np.max(np.abs(target_array))
            random_phase_target = np.random.uniform(-1, 1, size=8) + 1j * np.random.uniform(-1, 1, size=8)
            target_array = target_array * random_phase_target[:, None, None]
            target_array_close = np.isclose(target_array, 0).astype(int)
            target_array_outside = (target_array_close * mask_int).astype(bool)
            target_array[target_array_outside] = 0

            if self.transform_type_target is None:
                target_array = self.transform_complex(target_array, stack_ax=self.stack_ax)
            else:
                target_array = self.transform_complex(target_array,
                                                      transform_type=self.transform_type_target,
                                                      stack_ax=self.stack_ax)

        # Put data in complex couples subsequently
        x = input_array
        y = target_array
        # x = np.moveaxis(input_array, -1, 0).T.reshape((256, 256, -1)).T
        # y = np.moveaxis(target_array, -1, 0).T.reshape((256, 256, -1)).T

        x_tens = torch.from_numpy(x.copy()).float()
        y_tens = torch.from_numpy(y.copy()).float()

        n_c = target_array.shape[0]
        mask_array = np.tile(mask_array, (n_c, 1, 1))
        mask_tensor = torch.from_numpy(mask_array).float()

        return {'input': x_tens, 'target': y_tens, 'mask': mask_tensor}


class DataSetSurvey2B1_single(data_gen.DatasetGenericComplex):
    def __init__(self, ddata, input_shape, target_shape=None, batch_perc=0.010,
                 shuffle=True, dataset_type='train', file_ext='npy', **kwargs):

        input_args = {k: v for k, v in locals().items() if (k !='self') and (k !='__class__')}
        super().__init__(**input_args)

        self.model_choice = kwargs.get('model_choice', None)
        self.coil_choice = kwargs.get('coil_choice', None)
        self.coil_choice = kwargs.get('return_coil', None)

    def __getitem__(self, index):
        """Generate one batch of data"""
        im_y, im_x  = self.img_target_shape[-2:]
        i_file = self.file_list[index]
        input_file = os.path.join(self.input_dir, i_file)

        # Will be of shape 8, 8, X, Y
        input_array = np.load(input_file)
        n_c, n_c, n_y, n_x = input_array.shape
        # Chosen to NOT sample from a range.. but always ni the center.
        # Might re-do this, but with a fixed range from the center
        # y0 = np.random.randint(0, n_y - im_y + 1)
        # x0 = np.random.randint(0, n_x - im_x + 1)
        y0 = (n_y - im_y) // 2
        x0 = (n_x - im_x) // 2
        n_y_range = np.arange(im_y) + y0
        n_x_range = np.arange(im_x) + x0

        # Subset the matrix to the chosen size..
        input_array = np.take(input_array, n_y_range, axis=-2)
        input_array = np.take(input_array, n_x_range, axis=-1)

        if self.coil_choice is not None:
            sel_coil = self.coil_choice
        else:
            sel_coil = np.random.randint(n_c)

        temp_input = np.take(input_array, sel_coil, axis=1).sum(axis=0)  # Select a receive channel

        input_scaling = np.max(np.abs(temp_input))
        temp_input = temp_input / input_scaling
        temp_input = self.transform_complex(temp_input, stack_ax=self.stack_ax)
        if self.input_is_output:
            temp_target = np.copy(temp_input)
        else:
            temp_target = np.take(input_array, sel_coil, axis=0).sum(axis=0)  # Select a transmit channel
            temp_target = temp_target / input_scaling
            temp_target = self.transform_complex(temp_target, stack_ax=self.stack_ax)

        x = temp_input
        y = temp_target

        return torch.tensor(x).float(), torch.tensor(y).float()


class DataSetSurvey2B1_all(data_gen.DatasetGenericComplex):
    def __init__(self, ddata, input_shape, target_shape=None, batch_perc=0.010, shuffle=True,
                 dataset_type='train', file_ext='npy', **kwargs):

        input_args = {k: v for k, v in locals().items() if (k !='self') and (k !='__class__')}
        super().__init__(**input_args)
        self.model_choice = kwargs.get('model_choice', None)
        self.relative_phase = kwargs.get('relative_phase', False)
        self.masked = kwargs.get('masked', False)
        self.fourier_transform = kwargs.get('fourier_transform', False)

        smooth_trans = htransform.TransformSmooth(3, prob=False)
        self.transform_obj_list = [smooth_trans]   # For now only this...

    def __getitem__(self, index):
        im_y, im_x = self.img_input_shape[-2:]
        i_file = self.file_list[index]
        input_file = os.path.join(self.input_dir, i_file)

        # Will be of shape 8, 8, X, Y
        input_array = np.load(input_file)
        n_c, n_c, n_y, n_x = input_array.shape
        # Chosen to NOT sample from a range.. but always ni the center.
        # Might re-do this, but with a fixed range from the center
        # y0 = np.random.randint(0, n_y - im_y + 1)
        # x0 = np.random.randint(0, n_x - im_x + 1)
        y0 = (n_y - im_y) // 2
        x0 = (n_x - im_x) // 2
        n_y_range = np.arange(im_y) + y0
        n_x_range = np.arange(im_x) + x0

        # select a submatrix
        input_array = np.take(input_array, n_y_range, axis=-2)
        input_array = np.take(input_array, n_x_range, axis=-1)

        if self.masked:
            abs_summed = np.abs(input_array).sum(axis=0).sum(axis=0)
            treshhold = np.max(abs_summed) * 0.1
            c_tresh = (abs_summed > treshhold).astype(int)
            n_mask = 32
            kernel = np.ones((n_mask, n_mask)) / n_mask ** 2
            smooth_mask = scipy.signal.convolve2d(c_tresh, kernel, mode='same', boundary='symm')

        # Relative phase to receive array
        if self.relative_phase:
            sel_coil = np.random.randint(n_c)
            # print('Using relative phase to coil ', sel_coil)
            phi_rel = np.angle(input_array[:, sel_coil])
            complex_phi = np.exp(-1j * phi_rel)
            input_array = input_array * complex_phi[:, np.newaxis]

        # Below gives us Rx interference...
        random_phase_input = np.random.uniform(-1, 1, size=8) + 1j * np.random.uniform(-1, 1, size=8)
        random_phase_input = random_phase_input / np.abs(random_phase_input)
        temp_input = np.einsum("r, rcxy->cxy", random_phase_input, input_array)
        # temp_input = input_array.sum(axis=0)  # Sum over the axis that contains all the Rx channels

        # Here should be the conversion to kspace..
        if self.fourier_transform:
            print(temp_input.shape)
            temp_input = harray.transform_kspace_to_image_fftn(temp_input, dim=(-2, -1))

        scale_input = np.max(np.abs(temp_input))
        temp_input = temp_input / scale_input
        # Transform object to smoother version...
        # seed = np.random.randint(2147483647)
        # np.random.seed(seed)  # Used so that the probs are all executed the same
        # for i_coil in range(n_c):
        #     for i_trans in self.transform_obj_list:
        #         temp_input[i_coil] = i_trans(temp_input[i_coil])

        temp_input = self.transform_complex(temp_input, stack_ax=self.stack_ax)
        if self.input_is_output:
            # If input is equal to output.. we need to do half of the operations..
            temp_target = np.copy(temp_input)
        else:
            # Below gives us Tx interference...
            random_phase_target = np.random.uniform(-1, 1, size=8) + 1j * np.random.uniform(-1, 1, size=8)
            random_phase_target = random_phase_target / np.abs(random_phase_target)
            temp_target = np.einsum("c, rcxy->rxy", random_phase_target, input_array)
            # temp_target = input_array.sum(axis=1)  # Sum over the axis that contains all the Tx channels

            if self.fourier_transform:
                temp_target = harray.transform_kspace_to_image_fftn(temp_target, dim=(-2, -1))

            # Schaling eerst.. anders exp-overflow -> dit was voor relative phases vooral...
            temp_target = temp_target / scale_input  # np.max(np.abs(temp_target))

            # np.random.seed(seed)  # Used so that the probs are all executed the same
            # for i_coil in range(n_c):
            #     for i_trans in self.transform_obj_list:
            #         temp_target[i_coil] = i_trans(temp_target[i_coil])

            # Act out the same transformation on input as on target.
            if self.transform_type_target is None:
                temp_target = self.transform_complex(temp_target, stack_ax=self.stack_ax)
            else:
                temp_target = self.transform_complex(temp_target, self.transform_type_target, stack_ax=self.stack_ax)

        input_tensor = torch.tensor(temp_input).float()
        target_tensor = torch.tensor(temp_target).float()
        mask_tensor = torch.as_tensor(smooth_mask).float()
        return {'input': input_tensor, 'target': target_tensor, 'mask': mask_tensor}


class DataSetSurvey2B1_all_svd(data_gen.DatasetGenericComplex):
    def __init__(self, ddata, input_shape, target_shape=None, batch_perc=0.010, shuffle=True,
                 dataset_type='train', file_ext='npy', **kwargs):

        input_args = {k: v for k, v in locals().items() if (k !='self') and (k !='__class__')}
        super().__init__(**input_args)
        self.model_choice = kwargs.get('model_choice', None)
        self.relative_phase = kwargs.get('relative_phase', False)
        self.masked = kwargs.get('masked', False)

    def __getitem__(self, index):
        im_y, im_x = self.img_input_shape[-2:]
        i_file = self.file_list[index]

        input_file = os.path.join(self.input_dir, i_file)
        target_file = os.path.join(self.target_dir, i_file)

        # Will be of shape 8, X, Y, complex valued..
        input_array = np.load(input_file)
        target_file = re.sub('left', 'right', target_file)
        target_array = np.load(target_file)
        n_c, n_y, n_x = input_array.shape
        # Chosen to NOT sample from a range.. but always ni the center.
        # Might re-do this, but with a fixed range from the center
        # y0 = np.random.randint(0, n_y - im_y + 1)
        # x0 = np.random.randint(0, n_x - im_x + 1)
        y0 = (n_y - im_y) // 2
        x0 = (n_x - im_x) // 2
        n_y_range = np.arange(im_y) + y0
        n_x_range = np.arange(im_x) + x0

        # select a submatrix
        input_array = np.take(input_array, n_y_range, axis=-2)
        temp_input = np.take(input_array, n_x_range, axis=-1)

        target_array = np.take(target_array, n_y_range, axis=-2)
        temp_target = np.take(target_array, n_x_range, axis=-1)

        temp_input = self.transform_complex(temp_input, stack_ax=self.stack_ax)

        if self.input_is_output:
            # If input is equal to output.. we need to do half of the operations..
            temp_target = np.copy(temp_input)
        else:
            if self.transform_type_target is None:
                temp_target = self.transform_complex(temp_target, stack_ax=self.stack_ax)
            else:
                temp_target = self.transform_complex(temp_target, transform_type=self.transform_type_target, stack_ax=self.stack_ax)

        if self.masked:
            temp_input[temp_input == 1] = 0
            temp_target[temp_target == 1] = 0

        return torch.tensor(temp_input).float(), torch.tensor(temp_target).float()


if __name__ == "__main__":
    dir_data = '/home/bugger/Documents/data/test_clinic_registration/registrated_h5'

    datagen_obj = DataGeneratorBmin2Bplus(dir_data, file_ext='h5', input_shape=(2,2), dataset_type='test')
    cont = datagen_obj.__getitem__(2)
    hplotc.SlidingPlot(cont['input'])
    hplotc.SlidingPlot(cont['target'])
