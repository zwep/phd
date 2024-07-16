
import numpy as np
import data_generator.Generic as data_gen
import helper.plot_class as hplotc
import torch
import matplotlib.pyplot as plt
import helper.array_transf as harray
import skimage.transform as sktransform
import tooling.shimming.b1shimming_single as mb1_single
import helper.misc as hmisc
import os
import h5py
import scipy.ndimage
import re


class DataGeneratorCardiacSegment(data_gen.DatasetGenericComplex):
    # This guy is needed to get a cardiac segmentatino with inhomogneeity induced images
    n_classes = 3
    def __init__(self, ddata, input_shape=None, target_shape=None,
                 shuffle=True, dataset_type='train', file_ext='npy', transform=None, **kwargs):
        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, transform=transform, **kwargs)

        self.target_type = kwargs.get('target_type', None)
        self.presentation_mode = kwargs.get('presentation_mode', False)
        self.random_mask = kwargs.get('random_mask', True)
        if self.target_type is None:
            print('Watch out, no target type is set.')

    def shim_and_apply(self, complex_array, mask):
        shimming_obj = mb1_single.ShimmingProcedure(complex_array, mask, relative_phase=True,
                                                    str_objective='b1', debug=self.debug)
        x_opt, final_value = shimming_obj.find_optimum()
        complex_array_shimmed = harray.apply_shim(complex_array, cpx_shim=x_opt)
        return complex_array_shimmed

    def __getitem__(self, index):
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']
        target_clean_dir = self.container_file_info[sel_dataset]['target_dir'] + "_clean"
        target_segm_dir = self.container_file_info[sel_dataset]['target_dir'] + "_segmentation"
        mask_dir = self.container_file_info[sel_dataset]['mask_dir']
        index = index % len(file_list)
        i_file = file_list[index]

        # Load b1 minus/plus/mask/"rho"/segmentation
        b1_minus_file = os.path.join(input_dir, i_file)
        b1_plus_file = os.path.join(target_dir, i_file)
        mask_file = os.path.join(mask_dir, i_file)
        target_clean_file = os.path.join(target_clean_dir, i_file)
        target_segm_file = os.path.join(target_segm_dir, i_file)

        # Load mask/rho/segmentation
        mask_array = hmisc.load_array(mask_file)
        rho_array = hmisc.load_array(target_clean_file)
        segm_array = hmisc.load_array(target_segm_file)
        cardiac_mask = (segm_array > 0).astype(int)

        # Prepare b1 minus
        b1_minus_array_cpx = hmisc.load_array(b1_minus_file)
        b1_minus_array = b1_minus_array_cpx[0] + 1j * b1_minus_array_cpx[1]
        # Choose a random coil to get data from
        n_c = b1_minus_array.shape[0]
        sel_coil = np.random.randint(n_c)
        phi_rel = np.angle(b1_minus_array[sel_coil])
        b1_minus_array = b1_minus_array * np.exp(-1j * phi_rel)
        b1_minus_array = harray.scale_minmax(b1_minus_array, is_complex=True)
        # -> Apply shimming procedure
        if self.random_mask:
            cardiac_mask = harray.create_random_center_mask(b1_minus_array.shape,
                                                            random=True,
                                                            mask_fraction=0.07)
        b1_minus_array_shimmed = self.shim_and_apply(b1_minus_array, cardiac_mask)


        # Prepare b1 plus
        b1_plus_array_cpx = hmisc.load_array(b1_plus_file)
        b1_plus_array = b1_plus_array_cpx[0] + 1j * b1_plus_array_cpx[1]
        n_c = b1_plus_array.shape[0]
        sel_coil = np.random.randint(n_c)
        phi_rel = np.angle(b1_plus_array[sel_coil])
        b1_plus_array = b1_plus_array * np.exp(-1j*phi_rel)
        b1_plus_array = harray.scale_minmax(b1_plus_array, is_complex=True)
        # --> Apply shimming procedire
        if self.random_mask:
            cardiac_mask = harray.create_random_center_mask(b1_plus_array.shape,
                                                            random=True,
                                                            mask_fraction=0.07)
        b1_plus_array_shimmed = self.shim_and_apply(b1_plus_array, cardiac_mask)
        # --> Apply linear scale
        target_flip_angle = np.deg2rad(10)
        # --> Use a (binary) mask to determine the current average signal
        x_sub = b1_plus_array_shimmed * cardiac_mask
        # --> Scale the mean over the cardiac mask to flip angle
        x_mean = np.abs(x_sub).sum() / np.sum(cardiac_mask)
        flip_angle_map = np.abs(b1_plus_array_shimmed) / x_mean * target_flip_angle
        b1_plus_signal = np.sin(flip_angle_map)

        # Prem segmentation array -> Convert integer valued to multi channel binary image
        xy_shape = segm_array.shape[-2:]
        segm_bin_array = np.zeros((self.n_classes,) + xy_shape)
        for i_class in range(1, self.n_classes + 1):
            segm_bin_array[i_class - 1] = segm_array == i_class

        segm_bin_array = segm_bin_array.astype(int)

        # Create the bias field array...
        bias_field_array = np.abs(b1_minus_array_shimmed) * b1_plus_signal

        if self.debug and self.debug_display_counter == 0:
            print("Shape of rho array ", rho_array.shape)
            print("Shape of b1 min array ", b1_minus_array_shimmed.shape)
            print("Shape of b1 plus array ", b1_plus_array_shimmed.shape)

        # Lets try to rotate the B1 field...
        b1_mask = np.abs(b1_minus_array_shimmed) > 0
        rho_mask = None
#        temp1 = scipy.ndimage.rotate(x, angle=angle1, axes=(-2, -1))
        # Create input array / tensor
        input_array = rho_array * b1_minus_array_shimmed * b1_plus_signal
        input_array = harray.scale_minmax(input_array, is_complex=True)
        input_array = self.transform_complex(input_array, stack_ax=self.stack_ax)
        input_array = harray.correct_mask_value(input_array, mask_array)

        # Create target array / tensor
        if self.target_type == 'rho':
            target_array = rho_array
        elif self.target_type == 'biasfield':
            target_array = bias_field_array
        elif self.target_type == 'segmentation':
            target_array = segm_bin_array
        else:
            print('We have an invalid target type', self.target_type)
            print("Choose from: rho, biasfield, segmentation")
            target_array = None

        target_array = harray.scale_minmax(target_array)
        if self.transform_type_target is None:
            target_array = self.transform_complex(target_array, stack_ax=self.stack_ax)
        else:
            target_array = self.transform_complex(target_array,
                                                  transform_type=self.transform_type_target,
                                                  stack_ax=self.stack_ax)

        target_array = harray.correct_mask_value(target_array, mask_array)

        # Here we have the resize transform....
        if self.transform_resize:  # and self.dataset_type == 'train':
            # This can probably be done prettier.. but yeah..
            resize_shape = self.resize_list[self.resize_index]
            input_array = self.perform_resize(input_array, resize_shape=resize_shape)
            target_array = self.perform_resize(target_array, resize_shape=resize_shape)
            mask_array = self.perform_resize(mask_array, resize_shape=resize_shape)

        target_tensor = torch.from_numpy(target_array).float()
        input_tensor = torch.from_numpy(input_array).float()
        mask_tensor = torch.from_numpy(mask_array[None]).float()

        if self.debug and self.debug_display_counter == 0:
            print('Shape of input data')
            print('Input ', input_tensor.shape)
            print('Mask ', mask_tensor.shape)
            print('target ', target_tensor.shape)

        container_dict = {'input': input_tensor, 'target': target_tensor, 'mask': mask_tensor, 'file_name': i_file}

        if self.presentation_mode:
            presentation_dict = {'target_clean': rho_array,
                                 'b1p_shim': b1_plus_array_shimmed,
                                 'b1p_signal': b1_plus_signal,
                                 'b1p_array': b1_plus_array,
                                 'b1m_array': b1_minus_array,
                                 'b1m_shim': b1_minus_array_shimmed}
            container_dict.update(presentation_dict)

        self.debug_display_counter += 1

        # Add more Transformations only when we are training
        # This is used so that we have the same thing on input and target
        # Useful for example when rotating stuff
        random_seed_fixed = np.random.randint(123456789)
        if self.transform_compose is not None and self.dataset_type == 'train':
            for key, value in container_dict.items():
                if self.debug:
                    print('Applying data augmentation to', key)
                torch.manual_seed(random_seed_fixed)
                temp = value
                for i_transform in self.transform_compose.transforms:
                    transform_name = i_transform._get_name()
                    print('\t Transform', transform_name)
                    # Only perform Random Erasing / Gaussian blur on the input
                    # (Or: when we have something unequal to input, go to the next iteration)
                    only_on_input_keys = ["GaussianBlur", "RandomErasing"]

                    only_on_input_bool = [x == transform_name for x in only_on_input_keys]
                    if any(only_on_input_bool) and (key != 'input'):
                        continue

                    temp = i_transform(temp)

                container_dict[key] = temp

        return container_dict
