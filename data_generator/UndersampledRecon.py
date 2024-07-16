

"""
Here we have the data generators for the Survey 2 B1 maps

either single channel.. or all channels..
"""

import torch.utils.data
import torch.utils.data
import h5py
import os
import re
import data_generator.Generic as data_gen
import numpy as np
import nibabel
import helper.nufft_recon as hnufft
import skimage.transform as sktransf
import sigpy
import sigpy.mri
import helper.array_transf as harray
import skimage.transform as sktransform
import helper.misc as hmisc
#
# class DataGeneratorRadiallyUndersampled(data_gen.DatasetGenericComplex):
#     def __init__(self, ddata, input_shape=None, target_shape=None,
#                  shuffle=True, dataset_type='train', file_ext='npy', transform=None, **kwargs):
#         super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
#                          dataset_type=dataset_type, file_ext=file_ext, transform=transform, **kwargs)
#
#         self.p_undersample = kwargs.get('p_undersample', 100)
#         # With associated DCF
#         self.ovs = 1.25
#         self.width = 6
#         self.selected_dataset = None
#     def __getitem__(self, item):
#         # Here we want to load an image.. of any extenion.. and undersampled it radially
#         # The options we have are...
#         # Cartesian data [npy] (ncard, nx ,ny)
#         # Radial data [npy] (ncoil, ncard, nx, ny)
#         # MM Segment [nii.gz] (nx, ny, nloc) -> Hierdoor kan de 3 dimensie truuc niet meer. Wel de extensie
#         # In all cases we simply want to get back the fully sampled one.
#         # But how are we going to do that with multiple coils..? abs sum complex?
#
#         container = {}
#         return container


class DataGeneratorSemireal(data_gen.DatasetGenericComplex):
    def __init__(self, ddata, input_shape=None, target_shape=None,
                 shuffle=True, dataset_type='train', file_ext='npy', transform=None, **kwargs):
        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, transform=transform, **kwargs)

    def __getitem__(self, index):
        """Generate one batch of data"""
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']
        mask_dir = self.container_file_info[sel_dataset]['mask_dir']
        i_file = file_list[index]

        re_obj = re.findall("(\w+)_to_(\w+).h5", i_file)
        if re_obj is not None:
            flavio_name, prostate_name = re_obj[0]

        # Creating input File object
        input_file = os.path.join(input_dir, i_file)
        input_h5_obj = h5py.File(input_file, 'r')['data']

        # Creating mask File object
        mask_file = os.path.join(mask_dir, prostate_name + '.h5')
        mask_h5_obj = h5py.File(mask_file, 'r')['data']

        # Creating target File object
        if self.optional_target_appendix == "_clean":
            target_file = os.path.join(target_dir, prostate_name + '.h5')
        else:
            target_file = os.path.join(target_dir, i_file)
        target_h5_obj = h5py.File(target_file, 'r')['data']

        # Loading input file
        if input_h5_obj.ndim == 5:
            # This happens if we take the /input folder_15
            # Now we have the order... traj, phase, slice, n_y, n_x
            n_traj, n_phase, n_slice, _, _ = input_h5_obj.shape

            sel_phase = np.random.randint(n_phase)
            sel_trajectory = np.random.randint(n_traj)
            sel_slice = np.random.randint(n_slice)

            input_array = np.array(input_h5_obj[sel_trajectory, sel_phase, sel_slice])

        elif input_h5_obj.ndim == 4:
            # This happens if we take the /input folder
            n_phase, n_slice, _, _ = input_h5_obj.shape

            sel_phase = np.random.randint(n_phase)
            sel_slice = np.random.randint(n_slice)

            input_array = np.array(input_h5_obj[sel_phase, sel_slice])
        else:
            sel_phase = 0
            sel_slice = 0
            input_array = None
            print('ERROR: Unkown shape of input object')

        # Loading target file
        if self.optional_target_appendix == "_clean":
            target_array = np.array(target_h5_obj[sel_slice]).astype(np.float)
            target_array = harray.scale_minmax(target_array)
        else:
            target_array = np.array(target_h5_obj[sel_phase, sel_slice])

        # Transforming input file to stacked version
        input_array = self.transform_complex(input_array, transform_type=self.transform_type_input, stack_ax=self.stack_ax)

        # Transforming target file to stacked version
        if self.transform_type_target is None:
            target_array = self.transform_complex(target_array, stack_ax=self.stack_ax)
        else:
            target_array = self.transform_complex(target_array, transform_type=self.transform_type_target, stack_ax=self.stack_ax)

        # Selecting correct slice for mask..
        mask_array = np.array(mask_h5_obj[sel_slice])
        n_c = target_array.shape[0]
        # Copy the Mask array such that it corresponds with all n_c
        mask_array = np.tile(mask_array, (n_c, 1, 1))

        input_tensor = torch.as_tensor(input_array).float()
        target_tensor = torch.as_tensor(target_array).float()
        mask_tensor = torch.as_tensor(mask_array).float()
        container_dict = {'input': input_tensor, 'target': target_tensor, 'mask': mask_tensor}

        return container_dict


class DataGeneratorUndersampledRadial(data_gen.DatasetGenericComplex):
    """
    THis is simply.. undersample data radially and fit to its fully sampled version
    Input can be fully sampled radial
    But can also be cartesian....??
    """
    def __init__(self, ddata, input_shape=None, target_shape=None,
                 shuffle=True, dataset_type='train', file_ext='npy', transform=None, **kwargs):
        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, transform=transform, **kwargs)

        self.p_undersample = kwargs.get('p_undersample', 100)
        # With associated DCF
        self.resize_list = [(256, 256)]
        self.resize_index = 0
        self.ovs = 1.25
        self.width = 6
        self.selected_dataset = None

    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.selected_dataset is None:
            self.selected_dataset = np.random.choice(self.n_datasets)

        file_list = self.container_file_info[self.selected_dataset]['file_list']
        input_dir = self.container_file_info[self.selected_dataset]['input_dir']
        index = index % len(file_list)
        i_file = file_list[index]
        input_file = os.path.join(input_dir, i_file)
        loaded_array = hmisc.load_array(input_file)
        if loaded_array.ndim == 3:
            # Now we probably deal with cartesian unfolded data. This has no coil dimension.
            # Thus we add a new axis in front of it
            loaded_array = loaded_array[np.newaxis]

        # We could select only the last 8 coils if we want...
        # Ideally we want to have a subset of this or something like that.
        # loaded_array = loaded_array[-8:]
        n_loc, n_card = loaded_array.shape[:2]
        sel_loc = np.random.randint(0, n_loc)
        sel_card = np.random.randint(0, n_card)
        card_array = loaded_array[sel_loc, sel_card]
        img_shape = card_array.shape[-2:]

        # This is almost the same over all the images because of a weird circle...
        x_size, y_size = img_shape
        x_range = np.linspace(-x_size//2, x_size//2, x_size)
        y_range = np.linspace(-y_size // 2, y_size // 2, y_size)
        X, Y = np.meshgrid(x_range, y_range)
        mask_array = np.sqrt(X ** 2 + Y ** 2) <= x_size//2

        # Define dimensions radial spokes
        n_spokes = n_points = max(img_shape)
        n_spokes = int(n_spokes * np.pi/2)
        n_undersample = int((self.p_undersample / 100) * n_spokes)
        # Define trajectory..
        trajectory_radial = sigpy.mri.radial(coord_shape=(n_spokes, n_points, 2), img_shape=img_shape)
        trajectory_radial = trajectory_radial.reshape(-1, 2)
        # We might remove this one..? Because it is so generic and repetitive over all the spokes
        dcf = np.sqrt(trajectory_radial[:, 0] ** 2 + trajectory_radial[:, 1] ** 2)

        # Define undersampled trajectory, the same for ALL the coils
        undersampled_trajectory = np.array(np.split(trajectory_radial, n_spokes))
        # We selecteren hier indices van de lijnen die we WEG willen hebben
        # Dus bij undersampled trajectory worden er n - n_undersampled lijnen op 'null' gezet
        # Zo behouden n_undersampled lijnen hun data
        random_lines = np.random.choice(range(n_spokes), size=(n_spokes - n_undersample), replace=False)
        undersampled_trajectory[random_lines] = None
        undersampled_trajectory = undersampled_trajectory.reshape(-1, 2)

        if self.debug:
            print('Number of spokes/points ', n_spokes, '/', n_points)
            print('Number of undersample spokes ', n_undersample)
            print('Min/Max traj', trajectory_radial.min(), trajectory_radial.max())
            print('Min/Max us traj', undersampled_trajectory.min(), undersampled_trajectory.max())

        input_array = []
        # I could make a choice on WHICH coils we take..
        # We can take all.. we can also vary them
        # for i_coil in card_array[-8:]:
        temp_kspace = sigpy.nufft(card_array, coord=undersampled_trajectory, width=self.width, oversamp=self.ovs)
        input_array = sigpy.nufft_adjoint(temp_kspace * dcf, coord=undersampled_trajectory, oshape=img_shape, width=self.width, oversamp=self.ovs)

        input_array = np.array(input_array)
        # is it always complex valued? No. Not in the case of nii.gz data
        input_array = harray.scale_minpercentile(input_array, q=99, is_complex=True)
        input_array = harray.scale_minmax(input_array, is_complex=True)
        input_array = self.transform_complex(input_array, transform_type=self.transform_type_input, stack_ax=self.stack_ax)

        # Target is the fully sampled example
        target_array = card_array
        # Transforming target file to stacked version
        if self.transform_type_target is None:
            target_array = self.transform_complex(target_array, stack_ax=self.stack_ax)
        else:
            target_array = self.transform_complex(target_array, transform_type=self.transform_type_target, stack_ax=self.stack_ax)

        target_array = harray.scale_minpercentile(target_array, q=99)
        target_array = harray.scale_minmax(target_array)

        # Here we have the resize transform....
        if self.transform_resize:
            # This can probably be done prettier.. but yeah..
            resize_shape = self.resize_list[self.resize_index]
            input_array = self.perform_resize(input_array, resize_shape=resize_shape)
            target_array = self.perform_resize(target_array, resize_shape=resize_shape)
            mask_array = self.perform_resize(mask_array, resize_shape=resize_shape)

        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(target_array).float()
        mask_tensor = torch.from_numpy(mask_array).float()[None]

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


class DataGeneratorUndersampledCartesian(data_gen.DatasetGenericComplex):
    """
    This one is created to

    Input: load high undersampled data (more heart phases)
        Before the training, the input is first pushed through a petrained model
    Target: cartesian data (even more heart phases)

    These input/target are kinda fixed because of the dimensions they have...
    """

    def __getitem__(self, index):
        """Generate one batch of data"""
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']
        index = index % len(file_list)
        i_file = file_list[index]

        input_array = np.load(os.path.join(input_dir, i_file))
        target_array = np.load(os.path.join(target_dir, i_file))
        if self.debug:
            print('Shape of input array', input_array.shape)
            print('Shape of target array', target_array.shape)

        n_coils, n_card_input, _, _ = input_array.shape
        n_card_target, _, _ = target_array.shape

        sel_card_input = np.random.randint(0, n_card_input)
        sel_card_target = int(sel_card_input * n_card_target / n_card_input)

        # Select only 8 coils for now.
        input_array = input_array[-8:, sel_card_input]
        target_array = target_array[sel_card_target]

        if self.debug:
            print('Shape of input array', input_array.shape)
            print('Shape of target array', target_array.shape)

        # Scale and convert array to tensor
        input_array = np.array(input_array)
        input_array = harray.scale_minpercentile(input_array, q=98, is_complex=True, axis=(-2, -1))
        input_array = harray.scale_minmax(input_array, is_complex=True, axis=(-2, -1))
        input_array = self.transform_complex(input_array, stack_ax=self.stack_ax)
        input_tensor = torch.from_numpy(input_array).float()

        # Scale and convert target array to tensor
        target_array = np.abs(target_array)
        target_array = harray.scale_minpercentile(target_array, q=98, axis=(-2, -1))
        target_array = harray.scale_minmax(target_array, axis=(-2, -1))

        if self.transform_type_target is None:
            target_array = self.transform_complex(target_array, stack_ax=self.stack_ax)
        else:
            target_array = self.transform_complex(target_array,
                                                  transform_type=self.transform_type_target,
                                                  stack_ax=self.stack_ax)

        target_tensor = torch.as_tensor(target_array).float()

        container_dict = {'input': input_tensor, 'target': target_tensor}
        return container_dict


class DataGeneratorUndersampledProcessed(data_gen.DatasetGenericComplex):
    """
    This one is created to

    Input: processed undersampled radial data
    Target: cartesian data (with equal hart phases)
    These input/target are kinda fixed because of the dimensions they have...
    """

    def __getitem__(self, index):
        """Generate one batch of data"""
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']
        index = index % len(file_list)
        i_file = file_list[index]

        input_array = np.load(os.path.join(input_dir, i_file))
        target_array = np.load(os.path.join(target_dir, i_file))

        if self.debug:
            print('Shape of input array', input_array.shape)
            print('Shape of target array', target_array.shape)

        # Sometimes... the number of cardiac phases is at the last dimension..
        n_card = input_array.shape[0]
        sel_card = np.random.randint(0, n_card)

        # Select only 8 coils for now.
        input_array = np.take(input_array, sel_card, axis=0)
        target_array = np.take(target_array, sel_card, axis=0)

        if self.debug:
            print('Shape of input array', input_array.shape)
            print('Shape of target array', target_array.shape)

        # Scale and convert array to tensor
        input_array = np.array(input_array)
        input_tensor = torch.from_numpy(input_array).float()[None]

        # Scale and convert target array to tensor
        target_array = np.abs(target_array)
        target_tensor = torch.as_tensor(target_array).float()[None]

        container_dict = {'input': input_tensor, 'target': target_tensor}
        return container_dict


if __name__ == "__main__":
    import importlib
    import helper.array_transf as harray
    import helper.plot_fun as hplotf
    import matplotlib.pyplot as plt

    # dir_data = '/media/bugger/MyBook/data/semireal/prostate_simulation_h5'
    #
    # # Normal
    # A = DataGeneratorSemireal(dir_data, dataset_type='test', debug=True)
    # # A.container_file_info[0]
    # container = A.__getitem__(1)
    # a = container['input'].numpy()
    # b = container['target'].numpy()
    # hplotf.plot_3d_list(a[None, ...], augm='np.real', title='input')
    # hplotf.plot_3d_list(b[None, ...], augm='np.real', title='output')
    # c = container['mask'].numpy()

    # Undersampled stuff
    import helper.plot_class as hplotc
    dir_data = '/media/bugger/MyBook/data/7T_data/radial_dataset_4ch'
    # dir_data = '/data/seb/unfolded_radial/radial_dataset_p2ch'
    np.random.seed(0)
    data_gen_obj = DataGeneratorUndersampledRadial(dir_data, input_is_output=True, p_undersample=20, shuffle=False,
                                                   complex_type='polar', transform_type='abs', transform_type_target='abssumcomplex', debug=True)

    index_specific_file = data_gen_obj.container_file_info[0]['file_list'].index('v9_06032021_1228313_17_3_4ch_radialV4.npy')
    res = data_gen_obj.__getitem__(index_specific_file)

    hplotc.ListPlot([res['mask'] * res['target'][None], np.abs(res['input'][::2]).mean(axis=0)], augm='np.abs', title=data_gen_obj.p_undersample)
