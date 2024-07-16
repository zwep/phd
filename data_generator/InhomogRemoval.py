
"""
Create a Data Generator that acts on complex multi channel input data..
And outputs... a simple magnitude image from nrrd files...
"""


import numpy as np
import data_generator.Generic as data_gen
import helper.plot_class as hplotc
import torch
import helper.array_transf as harray
import helper.misc as hmisc
import os
import h5py
import scipy.ndimage
import scipy.stats
import small_project.sinp3.signal_equation as signal_eq
import tooling.shimming.b1shimming_single as mb1_single
from loguru import logger
# import os
# import psutil
# pid = os.getpid()
# python_process = psutil.Process(pid)
# memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
# print('memory use:', memoryUse)
import os
DLOG = os.path.expanduser("~/logger")
logger.add(os.path.join(DLOG, 'InhomogRemoval.log'))


class DataGeneratorFullBodyProstate(data_gen.DatasetGenericComplex):
    """
    This is a helper class to set all the options for inhomogeneity removal

    There were more child-classes.. but most got obselete.
    """

    def __init__(self, ddata, input_shape=None, target_shape=None,
                 shuffle=True, dataset_type='train', file_ext='npy', transform=None, **kwargs):
        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, transform=transform, **kwargs)

        self.use_tx_shim = kwargs.get('use_tx_shim', False)
        # Deprecated
        # self.t1t2_type = kwargs.get('t1t2', 't2')
        # Deprecated
        # self.sel_slice = kwargs.get('sel_slice', None)
        self.alternative_input = kwargs.get('alternative_input', None)

        # Have bias field as target or the undisturbed image or b1p array or b1m...
        self.target_type = kwargs.get('target_type', None)
        self.mask_fraction = kwargs.get('mask_fraction', 0.07)
        self.relative_phase = kwargs.get('relative_phase', False)
        self.objective_shim = kwargs.get('objective_shim', 'signal_se')
        self.shim_data_path = kwargs.get('shim_data_path', None)
        self.flip_angle = kwargs.get('flip_angle', np.pi / 2)
        # Choosing an integer sets the SNR level to that..
        self.SNR_mode = kwargs.get('SNR_mode', 'random')
        # If True, then randomly decides if we transform the data to:
        # (128, 128), (256, 256), (512, 512), (1024, 1024)
        self.transform_resize = kwargs.get('transform_resize', False)
        # self.resize_list = [(128, 128), (256, 256), (512, 512)]
        # self.resize_list = [(128, 128)]
        self.resize_list = [(256, 256)]
        self.resize_index = 0

    def linear_scale_b1(self, x, mask=None, flip_angle=np.pi/2):
        if mask is None:
            mask = harray.create_random_center_mask(x.shape)
        # Use a (binary) mask to determine the average signal
        x_sub = x * mask
        # Mean over the masked area...
        x_mean = np.abs(x_sub).sum() / np.sum(mask)
        flip_angle_map = np.abs(x) / x_mean * flip_angle
        return flip_angle_map

    def get_scale_factor_center(self, x, mask=None):
        # Targeting to make the center a value of 1....
        if mask is None:
            mask = harray.create_random_center_mask(x.shape)
        # Use a (binary) mask to determine the average signal
        x_sub = x * mask
        # Mean over the masked area...
        x_scale_factor = np.abs(x_sub).sum() / np.sum(mask)
        return x_scale_factor

    def simple_signal_model(self, flip_angle_map):
        return signal_eq.get_t2_signal_simplified(flip_angle_map)

    def signal_model(self, flip_angle_map):
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3310288/
        # Musculoskeletal MRI at 3.0T and 7.0T: A Comparison of Relaxation Times and Image Contrast
        TR_se = 5000
        TE_se = 100
        T1_fat = 583
        T2_fat = 46
        T1_muscle = 1552
        T2_muscle = 23
        T1 = (T1_fat + T1_muscle) / 2
        T2 = (T2_fat + T2_muscle) / 2
        N_refocus = 15

        general_signal_se = signal_eq.get_t2_signal_general(flip_angle=flip_angle_map,
                                                            T1=T1, TE=TE_se, TR=TR_se, T2=T2, N=N_refocus,
                                                            beta=flip_angle_map * 2)
        # Taking the absolute values to make sure that values are above 0
        # general_signal_se = np.abs(general_signal_se)

        return general_signal_se


class DataGeneratorInhomogRemovalH5(DataGeneratorFullBodyProstate):
    """
    Simple child class which only implemeSnts the get_item function

    Idea is that more child-classes can be created for different datasets while maintaining the options easily
    """
    noise_cov_path = '/home/bugger/Documents/paper/inhomogeneity removal/noise_cov_matrix/noise_cov.npy'
    # if os.path.isfile(noise_cov_path):
    #     covariance_matrix = np.load(noise_cov_path)
    #     covariance_matrix = harray.scale_minmax(covariance_matrix, is_complex=True)
    # else:
    covariance_matrix = np.eye(8)

    def __getitem__(self, index):
        # This piece of code will be almost identical to the one below... but created such that
        # we get individual components of the data creation
        """Generate one batch of data"""
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']
        target_clean_dir = self.container_file_info[sel_dataset]['target_dir'] + "_clean"
        mask_dir = self.container_file_info[sel_dataset]['mask_dir']
        # ADD SHIM PATH
        shim_dir = os.path.join(os.path.dirname(input_dir), 'shimsettings')
        # index = index % len(file_list)
        i_file = file_list[index]
        logger.info(f'Selecting file {i_file}')

        """Define all the paths for the images that we want to load..."""
        b1_minus_file = os.path.join(input_dir, i_file)
        b1_plus_file = os.path.join(target_dir, i_file)
        mask_file = os.path.join(mask_dir, i_file)
        target_clean = os.path.join(target_clean_dir, i_file)

        logger.info(f'File definitions')
        logger.info(f'\t {b1_minus_file}')
        logger.info(f'\t {b1_plus_file}')
        logger.info(f'\t {mask_file}')
        logger.info(f'\t {target_clean}')

        """Load all the data with a random slice selection"""
        with h5py.File(target_clean, 'r') as h5_obj:
            max_slice = h5_obj['data'].shape[0]

        logger.info(f'Maximum amount of slices {max_slice}')
        # Optional code to slide over all the slices..
        # # Make sure that we only do this during test and cycle all items enabled
        # # It costs some extra time I guess.. so we dont want it on all the time..
        sel_slice = np.random.randint(max_slice)
        if self.cycle_all_items and self.dataset_type == 'test':
            logger.info('Cycling all items of the test generator')
            slice_count = self.container_file_info[sel_dataset]['slice_count']
            sel_slice = slice_count[i_file].pop()

        if self.center_slice and self.dataset_type == 'test':
            logger.info('Taking the center slice of the test generator')
            sel_slice = max_slice // 2

        logger.info(f'Selecting slice {sel_slice}')
        shim_file_name = i_file + f"_slice_{sel_slice}.txt"
        shim_file_path = os.path.join(shim_dir, shim_file_name)

        with h5py.File(target_clean, 'r') as f:
            rho_array = np.array(f['data'][sel_slice])

        logger.info(f'Loaded rho array')
        # Scale it from uint to float 0..1
        rho_array = harray.scale_minmax(rho_array)

        if self.masked:
            with h5py.File(mask_file, 'r') as f:
                mask_array = np.array(f['data'][sel_slice])
            # # Needed otherwise we get weird banding stuff...
            # # Smoothing kernel is rather aribtrary
            # # This gives a mask that is a bit larger than we actually need, but that is fine IMO
            mask_array = scipy.ndimage.binary_fill_holes(harray.smooth_image(mask_array, n_kernel=8))
        #
        logger.info(f'Loaded mask array')
        with h5py.File(b1_plus_file, 'r') as f:
            b1_plus_array = np.array(f['data'][sel_slice])

        logger.info(f'Loaded B1+ array')
        b1_plus_array = b1_plus_array[0] + 1j * b1_plus_array[1]
        with h5py.File(b1_minus_file, 'r') as f:
            b1_minus_array = np.array(f['data'][sel_slice])
        b1_minus_array = b1_minus_array[0] + 1j * b1_minus_array[1]
        logger.info(f'Loaded B1- array')
        b1_minus_array = harray.scale_minmax(b1_minus_array, is_complex=True)
        b1_plus_array = harray.scale_minmax(b1_plus_array, is_complex=True)

        # # # #
        # THis is needed because the shims we use are optimized for this as well
        b1_plus_array = b1_plus_array * np.exp(-1j * np.angle(b1_plus_array[0]))

        if os.path.isfile(shim_file_path):
            logger.info(f'Selecting following shim file {shim_file_path}')
            # The procedure below requires a lot of CPU.. atleast with the Powell method as standard
            # Now we have changed to the BFG or something.. is a bit better I guess
            # However, I reverted to pre-computeed shim settings because that is easier..
            with open(shim_file_path, 'r') as f:
                list_shim_settings = f.readlines()

            n_shim_settings = len(list_shim_settings)
            sel_shim_set = np.random.choice(n_shim_settings)
            shim_set = list_shim_settings[sel_shim_set].strip()
            # In case we cannot convert the read line into a proper complex number
            # We revert back to the random shims
            try:
                x_opt = np.array([complex(x[1:-1]) for x in shim_set.split(',')])
                b1_plus_array = harray.apply_shim(b1_plus_array, cpx_shim=x_opt)
                if self.debug:
                    print('Used the following shim file ', shim_file_path)
                    print('Used the following shim settings ', x_opt)
            except:
                b1_plus_array = harray.apply_shim(b1_plus_array)
        else:
            logger.info(f'Applying a random shim')
            # # Misc
            # import tooling.shimming.b1shimming_single as mb1
            # center_mask = harray.create_random_center_mask(b1_plus_array.shape[-2:], random=False)
            # shim_proc = mb1.ShimmingProcedure(b1_plus_array, center_mask, relative_phase=True,
            #                                   str_objective='signal_se')
            # opt_shim, opt_value = shim_proc.find_optimum()
            # b1_plus_array = harray.apply_shim(b1_plus_array, cpx_shim=opt_shim)
            # # /Misc
            b1_plus_array = harray.apply_shim(b1_plus_array)

        logger.info(f'Scaling the center of the B1+ to 90 degrees FA')
        center_mask = harray.create_random_center_mask(b1_plus_array.shape, random=False, mask_fraction=self.mask_fraction)
        flip_angle_map = self.linear_scale_b1(b1_plus_array, mask=center_mask, flip_angle=self.flip_angle)
        b1_plus_array = self.signal_model(flip_angle_map=flip_angle_map)
        b1_plus_array = harray.scale_minmax(b1_plus_array)#, q=99, is_complex=False)
        logger.info(f'Scaling the B1- also to 1 I believe')
        scale_factor_b1min = self.get_scale_factor_center(b1_minus_array)
        b1_minus_array = b1_minus_array / scale_factor_b1min
        bias_field_array = np.abs(b1_minus_array).sum(axis=0) * (b1_plus_array)
        # # Define input array (bias field induced image..)
        input_array = rho_array * b1_minus_array * b1_plus_array
        # This might be necessary for the SNR calculations..
        # input_array = harray.scale_minmax(input_array, is_complex=True)
        # std_noise = 0.02
        SNR = np.random.randint(4, 15)
        if isinstance(self.SNR_mode, int):
            SNR = self.SNR_mode

        logger.info(f'Selecting the following SNR level {SNR}')
        # "echte" SNR ranged van 15..27 ong
        n_coils = input_array.shape[0]
        # sigma = (1 / (2 * np.sqrt(n_coils * SNR)))
        # sigma = np.random.uniform(0, 0.5 / 100)
        # var_sigma_min = 0
        # sigma = 0.5/100
        sigma = (2 / (4 - np.pi)) * (1 / (SNR ** 2 * (n_coils)))
        print('sigma ', sigma)

        imag_noise = np.random.multivariate_normal(np.zeros(n_coils),
                                                   np.eye(n_coils) * sigma,
                                                   size=rho_array.shape)
        real_noise = np.random.multivariate_normal(np.zeros(n_coils),
                                                   np.eye(n_coils) * sigma,
                                                   size=rho_array.shape)
        input_array = input_array + (real_noise + 1j * imag_noise).T
        logger.info(f'Added noise to the image')
        coil_order = np.arange(8)
        np.random.shuffle(coil_order)
        input_array = self.transform_complex(input_array[coil_order], stack_ax=self.stack_ax)
        #
        if self.target_type == 'rho':
            target_array = rho_array
        elif self.target_type == 'b1p_scaled':
            target_array = b1_plus_array
        elif self.target_type == 'b1m':
            target_array = b1_minus_array
        elif self.target_type == 'b1m_scaled':
            target_array = np.linalg.norm(b1_minus_array, axis=0)
        elif self.target_type == 'b1m_summed':
            target_array = harray.apply_shim(b1_minus_array)
        elif self.target_type == 'biasfield_coil':
            target_array = b1_minus_array * b1_plus_array
        elif self.target_type == 'biasfield':
            target_array = bias_field_array
        elif self.target_type == 'both':
            target_array = np.stack([rho_array, bias_field_array])
        else:
            print(f'Unknown target type: {self.target_type}')
            target_array = None

        logger.info(f'Selected the target type {self.target_type}')

        if self.transform_type_target is None:
            target_array = self.transform_complex(target_array, stack_ax=self.stack_ax)
        else:
            target_array = self.transform_complex(target_array,
                                                  transform_type=self.transform_type_target,
                                                  stack_ax=self.stack_ax)

        n_c = target_array.shape[0]
        # Everything is already real valued.. so we can use the original resize command.
        # This is needed
        if self.transform_resize:
            # This can probably be done prettier.. but yeah..
            resize_shape = self.resize_list[self.resize_index]
            input_array = self.perform_resize(input_array, resize_shape=resize_shape)
            target_array = self.perform_resize(target_array, resize_shape=resize_shape)
            mask_array = self.perform_resize(mask_array, resize_shape=resize_shape)


        # Copy the Mask array such that it corresponds with all n_c
        mask_array = np.tile(mask_array, (n_c, 1, 1))
        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(target_array).float()
        mask_tensor = torch.from_numpy(mask_array).float()

        container_dict = {'input': input_tensor, 'target': target_tensor, 'mask': mask_tensor}
        random_seed_fixed = np.random.randint(123456789)
        if self.transform_compose is not None:
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


class PreComputerShimSettings(DataGeneratorFullBodyProstate):
    counter = 0

    def __getitem__(self, index):
        """Generate one batch of data"""
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        target_dir = self.container_file_info[sel_dataset]['target_dir']

        mask_dir = self.container_file_info[sel_dataset]['mask_dir']
        # index = index % len(file_list)
        i_file = file_list[index]

        b1_plus_file = os.path.join(target_dir, i_file)
        mask_file = os.path.join(mask_dir, i_file)

        with h5py.File(b1_plus_file, 'r') as f:
            b1_plus_array = f['data']
            max_slice = b1_plus_array.shape[0]

        print('Processing file ', i_file)
        for sel_slice in range(0, max_slice):
            if sel_slice % (max_slice // 10) == 0:
                print('current slice', sel_slice, '/', max_slice, end='\r')

            with h5py.File(mask_file, 'r') as f:
                mask_array = np.array(f['data'][sel_slice])

            with h5py.File(b1_plus_file, 'r') as f:
                b1_plus_array = np.array(f['data'][sel_slice])

            b1_plus_array = b1_plus_array[0] + 1j * b1_plus_array[1]
            center_mask = harray.create_random_center_mask(b1_plus_array.shape, random=False, mask_fraction=self.mask_fraction)
            # Relative phase...
            b1_plus_array = b1_plus_array * np.exp(-1j * np.angle(b1_plus_array[0]))
            b1_plus_array = harray.correct_mask_value(b1_plus_array, mask_array)
            b1_plus_array = harray.scale_minmax(b1_plus_array, is_complex=True)

            shimming_obj = mb1_single.ShimmingProcedure(b1_plus_array, center_mask,
                                                        relative_phase=True,
                                                        str_objective=self.objective_shim,
                                                        debug=self.debug)

            x_opt, final_value = shimming_obj.find_optimum()
            shim_settings_file = os.path.join(os.path.dirname(target_dir), 'shimsettings', f'{i_file}_slice_{sel_slice}.txt')
            with open(shim_settings_file, 'a') as f:
                f.write(','.join([str(x) for x in x_opt]) + '\n')

            b1_shimmed = harray.apply_shim(b1_plus_array, cpx_shim=x_opt)
            flip_angle_map = self.linear_scale_b1(b1_shimmed, mask=center_mask, flip_angle=self.flip_angle)
            b1_signal = self.signal_model(flip_angle_map=flip_angle_map)

            mean_flip_angle = np.sum(flip_angle_map * center_mask) / np.sum(center_mask)
            if self.debug:
                print("\tMean flip angle in center mask ", mean_flip_angle)
                plot_obj = hplotc.ListPlot([np.abs(b1_shimmed) * (1 + center_mask), flip_angle_map, b1_signal], cbar=True)
                plot_obj.figure.savefig(f'/home/sharreve/local_scratch/examples_shim/image_{self.counter}.png')
                self.counter += 1
                hplotc.close_all()
        print('Done', end='\n\n')


class DataGeneratorCardiac(data_gen.DatasetGenericComplex):
    def __init__(self, ddata, input_shape=None, target_shape=None,
                 shuffle=True, dataset_type='train', file_ext='npy', transform=None, **kwargs):
        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, transform=transform, **kwargs)

        self.target_type = kwargs.get('target_type', None)
        if self.target_type is None:
            print('Watch out, no target type is set.')

        self.bins_expansion = kwargs.get('bins_expansion', 1)

    def __getitem__(self, index):
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']
        target_clean_dir = self.container_file_info[sel_dataset]['target_dir'] + "_clean"
        mask_dir = self.container_file_info[sel_dataset]['mask_dir']
        index = index % len(file_list)
        i_file = file_list[index]

        b1_minus_file = os.path.join(input_dir, i_file)
        b1_plus_file = os.path.join(target_dir, i_file)
        mask_file = os.path.join(mask_dir, i_file)
        target_clean_file = os.path.join(target_clean_dir, i_file)

        b1_minus_array = hmisc.load_array(b1_minus_file)[0]
        b1_plus_array = hmisc.load_array(b1_plus_file)[0]
        mask_array = hmisc.load_array(mask_file)[0]
        rho_array = hmisc.load_array(target_clean_file)[0]

        # print(b1_minus_array.shape)
        # print(b1_plus_array.shape)
        # print(mask_array.shape)
        # print(rho_array.shape)
        b1_plus_array_shimmed = harray.apply_shim(b1_plus_array)
        b1_minus_array_shimmed = harray.apply_shim(b1_minus_array)
        bias_field_array = np.abs(b1_minus_array_shimmed) * b1_plus_array_shimmed

        # Create mask array / tensor
        mask_tensor = torch.from_numpy(mask_array[None]).float()

        # Create input array / tensor
        input_array = rho_array * b1_minus_array * b1_plus_array_shimmed
        input_array = harray.scale_minmax(input_array, is_complex=True)
        input_array = self.transform_complex(input_array, stack_ax=self.stack_ax)
        input_array = harray.correct_mask_value(input_array, mask_array)

        input_tensor = torch.from_numpy(input_array).float()
        # Create target array / tensor
        if self.target_type == 'rho':
            target_array = rho_array
        elif self.target_type == 'biasfield':
            target_array = bias_field_array
        elif self.target_type == 'expansion':
            bias_field_array = harray.scale_minmax(bias_field_array)
            target_array = harray.split_array_fourier_basis(bias_field_array, n_bins=self.bins_expansion)
        else:
            print('We have an invalid target type', self.target_type)
            target_array = None

        target_array = harray.scale_minmax(target_array)
        if self.transform_type_target is None:
            target_array = self.transform_complex(target_array, stack_ax=self.stack_ax)
        else:
            target_array = self.transform_complex(target_array,
                                                  transform_type=self.transform_type_target,
                                                  stack_ax=self.stack_ax)

        target_array = harray.correct_mask_value(target_array, mask_array)
        target_tensor = torch.from_numpy(target_array).float()
        container_dict = {'input': input_tensor, 'target': target_tensor, 'mask': mask_tensor}

        return container_dict


if __name__ == "__main__":
    """
    Example of torchvision syntax
    
    generator_options.update({"transform": {"RandomAffine": {"scales": (0.5, 2),
                                                                 "degrees": (-10, 10, -10, 10, 0, 0),
                                                                 "translation": (-50, 50, -50, 50, 0, 0),
                                                                 "center": "image",
                                                                 "default_pad_value": 0,
                                                                 "isotropic": True}}})
                                                   
    "transform": {"RandomAffine": {"scales": [0.5, 2],
                 "degrees": [-10, 10, -10, 10, 0, 0],
                 "translation": [-50, 50, -50, 50, 0, 0],
                 "center": "image",
                 "default_pad_value": 0,
                 "isotropic": true}}
    """

    def visualize_data_object(gen, sel_index=0):
        container = gen.__getitem__(sel_index)
        input_cpx = container['input'][::2] + 1j * container['input'][1::2]
        target = container['target'].numpy()
        mask_np = container['mask'][0].numpy()
        som_input = np.abs(input_cpx).sum(axis=0).numpy()
        mask_resize = harray.resize_and_crop(mask_np, scale=1.2)
        mask_resize2 = harray.resize_and_crop(mask_np, scale=0.8)
        std_denom = np.std(som_input[mask_resize != 1])
        SNR_stuff = np.mean(som_input[mask_resize2 == 1]) / std_denom
        # print("Is this SNR? ", SNR_stuff)
        # Visualize created data
        fig_obj = hplotc.ListPlot([input_cpx, mask_np, som_input, target], augm='np.abs')
        fig_obj2 = hplotc.ListPlot([target, som_input / target], augm='np.abs', vmin=(0, 2), cbar=True)
        return fig_obj, fig_obj2

    dir_data = '/home/bugger/Documents/data/test_clinic_registration/registrated_h5'
    # shim_path = '/media/bugger/MyBook/data/simulated/transmit_flavio'
    shim_path = None

    generator_options = {"ddata": dir_data, "dataset_type": 'test', "complex_type": 'cartesian',
                                      "input_shape": (1, 256, 256),
                                      "use_tx_shim": False,
                                      "b1p_scaling": True,
                                      "debug": True,
                                      "shim_data_path": shim_path,
                                      "masked": True,
                                      "file_ext": "h5",
                                      "lower_prob": 0.0,
                                      "relative_phase": True,
                                      "objective_shim": 'flip_angle',
                                      "flip_angle": np.pi / 2}

    gen = DataGeneratorInhomogRemovalH5(target_type='rho', transform_type_target='real', **generator_options)
    temp = visualize_data_object(gen)
    sel_index = 0
    container = gen.__getitem__(sel_index)
    input_cpx = container['input'][::2] + 1j * container['input'][1::2]
    target = container['target'].numpy()
    mask_np = container['mask'][0].numpy()
    np.abs(input_cpx).numpy().max()
