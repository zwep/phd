import helper.plot_class as hplotc
import numpy as np
import data_generator.Generic as data_gen
import torch
import helper.array_transf as harray
import skimage.transform as sktransform
import os
import h5py

"""
Simply create a data generator... lets see what we get from this..
"""


class DataGeneratorPinnEfields(data_gen.DatasetGenericComplex):
    """
    This is a helper class to set all the options for inhomogeneity removal

    There were more child-classes.. but most got obselete.
    """
    def __init__(self, ddata, input_shape=None, target_shape=None,
                 shuffle=True, dataset_type='train', file_ext='h5', transform=None, **kwargs):
        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, transform=transform, **kwargs)
        self.target_type = kwargs.get('target_type', None)
    #
    def __getitem__(self, index):
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        sigma_dir = self.container_file_info[sel_dataset]['input_dir'] + '_sigma'
        eps_dir = self.container_file_info[sel_dataset]['input_dir'] + '_eps'
        B_dir = self.container_file_info[sel_dataset]['target_dir'] + '_B'
        E_dir = self.container_file_info[sel_dataset]['target_dir'] + '_E'
        #
        i_file = file_list[index]
        sigma_file = os.path.join(sigma_dir, i_file)
        eps_file = os.path.join(eps_dir, i_file)
        B_file = os.path.join(B_dir, i_file)
        E_file = os.path.join(E_dir, i_file)
        # Get the number of slices
        with h5py.File(sigma_file, 'r') as f:
            n_slice = f['data'].shape[0]
        #
        # Select a slice
        sel_slice = np.random.randint(n_slice)
        #
        # Load the data with the chosen slice..
        with h5py.File(sigma_file, 'r') as f:
            sigma_array = np.array(f['data'][sel_slice])
        #
        with h5py.File(eps_file, 'r') as f:
            eps_array = np.array(f['data'][sel_slice])
        #
        with h5py.File(B_file, 'r') as f:
            B_array = np.array(f['data'][sel_slice])
        #
        with h5py.File(E_file, 'r') as f:
            E_array = np.array(f['data'][sel_slice])
        #
        # Create a mask...?
        mask_array = harray.get_treshold_label_mask(eps_array)
        # mask_array = np.ones(sigma_array.shape)
        # Use only specific components..
        B_x_array = B_array[:, :, 0]
        B_y_array = B_array[:, :, 1]
        E_z_array = E_array[:, :, 2]
        # Get the shape for padding to 256, 256
        image_shape = B_x_array.shape
        delta_pad = 256 - image_shape[0]
        #
        # Get the location of the coil....
        y_ind, x_ind = np.unravel_index(np.abs(B_x_array).argmax(), np.abs(B_x_array).shape)
        coil_position = np.zeros(B_x_array.shape)
        coil_position[y_ind, x_ind] = 1
        #
        input_array = np.stack([eps_array * mask_array, sigma_array * mask_array, coil_position])
        input_array[np.isnan(input_array)] = 0
        input_array = np.pad(input_array,
                             ((0, 0), (delta_pad // 2, delta_pad // 2), (delta_pad // 2, delta_pad // 2)))
        if self.target_type == 'abs':
            # Cant apply the mask to the coil position.. otherwise that one gets lost..
            # Lets normalize over the first axis... see what that does
            # Normalizing like this is really stupid...
            # input_array = harray.scale_minmax(input_array)
            B_target = np.sqrt(np.abs(B_x_array) ** 2 + np.abs(B_y_array) ** 2)  # Zoiets??
            E_target = np.abs(E_z_array)
            # This target array has size (2, nx, ny)
            target_array = np.stack([B_target, E_target])
            # Normalizing like this is really stupid...
            target_array = harray.scale_minmax(target_array, is_complex=True) * mask_array[None]
            #
            target_array = np.pad(target_array, ((0, 0), (delta_pad//2, delta_pad//2), (delta_pad//2, delta_pad//2)))
        elif self.target_type == 'e_cpx':
            target_array = E_z_array
            target_array = np.pad(target_array, ((delta_pad//2, delta_pad//2), (delta_pad//2, delta_pad//2)))
        elif self.target_type == 'b_cpx':
            B_x_array = harray.scale_minmax(B_x_array, is_complex=True)
            B_y_array = harray.scale_minmax(B_y_array, is_complex=True)
            target_array = np.stack([B_x_array, B_y_array])
            #target_array = harray.scale_minmax(target_array)
            target_array = np.pad(target_array, ((0, 0), (delta_pad//2, delta_pad//2), (delta_pad//2, delta_pad//2)))
        elif self.target_type == 'e_b_cpx':
            target_array = np.stack([B_x_array, B_y_array, E_z_array])
            target_array = np.pad(target_array, ((0, 0), (delta_pad//2, delta_pad//2), (delta_pad//2, delta_pad//2)))
        else:
            input_array = None
            mask_array = None
            target_array = None
            print("Unkown target type: ", self.target_type)
        #
        if self.transform_type_target is None:
            target_array = self.transform_complex(target_array, stack_ax=self.stack_ax)
        else:
            target_array = self.transform_complex(target_array,
                                                  transform_type=self.transform_type_target,
                                                  stack_ax=self.stack_ax)
        #
        # Pad the mask array... to 256 shape
        mask_array = np.pad(mask_array, ((delta_pad // 2, delta_pad // 2), (delta_pad // 2, delta_pad // 2)))
        #
        input_tensor = torch.from_numpy(input_array.astype(float)).float()
        target_tensor = torch.from_numpy(target_array.astype(float)).float()
        # Add one dimensions to make it 3D...
        mask_tensor = torch.from_numpy(mask_array.astype(float)).float()[None]

        container_dict = {'input': input_tensor, 'target': target_tensor, 'mask': mask_tensor}

        random_seed_fixed = np.random.randint(123456789)
        if self.transform_compose is not None: # and self.dataset_type == 'train':
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


if __name__ == "__main__":
    ddata = '/local_scratch/sharreve/mri_data/pinn_fdtd'
    data_obj = DataGeneratorPinnEfields(ddata, dataset_type='train', target_type='b_cpx', transform_type_target='complex')
    container_obj = data_obj.__getitem__(0)
    input_array = container_obj['input'].numpy()
    target_array = container_obj['target'].numpy()
    mask_array = container_obj['mask'].numpy()
    input_array[np.isnan(input_array)] = 0
    fig_obj = hplotc.ListPlot([input_array, target_array * mask_array, mask_array], cbar=True)
    fig_obj.figure.savefig(os.path.join(ddata, 'example_data.png'))
    # Unpack the target array
    if data_obj.target_type == 'e_b_cpx':
        # target_array = np.stack([B_x_array, B_y_array, E_z_array])
        B_x_cpx = target_array[0] + 1j * target_array[1]
        B_y_cpx = target_array[2] + 1j * target_array[3]
        E_z_cpx = target_array[4] + 1j * target_array[5]
        print('Shape of Ez: ', E_z_cpx.shape)
        pad_mask = E_z_cpx == 0
