

"""

"""

import numpy as np
import data_generator.Generic as data_gen
import torch
import helper.array_transf as harray
import skimage.transform as sktransform
import os
import h5py


class DataGeneratorProstateWeighting(data_gen.DatasetGenericComplex):
    """
    This is a helper class to set all the options for inhomogeneity removal

    There were more child-classes.. but most got obselete.
    """

    def __init__(self, ddata, input_shape=None, target_shape=None,
                 shuffle=True, dataset_type='train', file_ext='h5', transform=None, **kwargs):
        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, transform=transform, **kwargs)

        self.alternative_input = kwargs.get('alternative_input', None)

        # If True, then randomly decides if we transform the data to:
        self.transform_resize = kwargs.get('transform_resize', False)
        # self.resize_list = [(256, 256), (512, 512)]
        # Leaving it at 256... for now...
        # Maybe somewhere I should allign the images even better.. Images are in the center of their own reference
        # But not of the 'global' stuff
        self.resize_list = [(256, 256)]

        self.resize_index = 0
        # I reviewed the training data.. and these should be omitted.
        # These are too dark (even the corrected version)
        patient_id = [6, 39, 42, 47]
        self.avoid_these_patients = [str(x) + "_MR.h5" for x in patient_id]

    def __getitem__(self, index):
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']
        mask_dir = self.container_file_info[sel_dataset]['mask_dir']

        index = index % len(file_list)
        i_file = file_list[index]
        # Make sure that we avoid any of these.
        # This is the
        while i_file in self.avoid_these_patients:
            index += 1
            index = index % len(file_list)
            i_file = file_list[index]

        file_name, ext = os.path.splitext(i_file)
        """Define all the paths for the images that we want to load..."""
        input_file = os.path.join(input_dir, i_file)
        target_file = os.path.join(target_dir, i_file)
        # Since we have specific input and target MASKS, we need to switch these too
        if self.switch_input_output:
            input_mask_file = os.path.join(mask_dir, file_name + '_target' + ext)
            target_mask_file = os.path.join(mask_dir, file_name + '_input' + ext)
        else:
            input_mask_file = os.path.join(mask_dir, file_name + '_input' + ext)
            target_mask_file = os.path.join(mask_dir, file_name + '_target' + ext)

        """Load all the data with a random slice selection"""
        with h5py.File(input_file, 'r') as h5_obj:
            input_max_slice = h5_obj['data'].shape[0]

        """Load all the data with a random slice selection"""
        with h5py.File(target_file, 'r') as h5_obj:
            target_max_slice = h5_obj['data'].shape[0]

        # Here we implement a linear relation between the input and target
        # Where we take into account that either the input or the target has more slices than the other
        min_slice = min(input_max_slice, target_max_slice)
        arg_min = np.argmin([input_max_slice, target_max_slice])
        max_slice = max(input_max_slice, target_max_slice)
        coefficient = max_slice / min_slice

        sel_slice_min = np.random.randint(min_slice)
        sel_slice_max = int(coefficient * sel_slice_min)

        if arg_min == 0:
            # Now the minimum slice comes from the input...
            sel_slice_input = sel_slice_min
            sel_slice_target = sel_slice_max
        else:
            # And here it comes from the target..
            sel_slice_target = sel_slice_min
            sel_slice_input = sel_slice_max

        if self.debug and self.debug_display_counter == 0:
            print('Status of shapes...')
            print('Input file ', input_file)
            print('Target file ', target_file, end='\n\n')
            print('Input mask file ', input_mask_file)
            print('Target mask file ', target_mask_file, end='\n\n')
            print('Input max slice ', input_max_slice)
            print('Target max slice ', target_max_slice, end='\n\n')
            print('Input sel slice ', sel_slice_input)
            print('Target sel slice ', sel_slice_target, end='\n\n')
            print('Coefficient ', coefficient)
            print('Min slice ', min_slice)

        with h5py.File(input_file, 'r') as f:
            input_array = np.array(f['data'][sel_slice_input])

        with h5py.File(input_mask_file, 'r') as f:
            input_mask_array = np.array(f['data'][sel_slice_input])[None]

        with h5py.File(target_file, 'r') as f:
            target_array = np.array(f['data'][sel_slice_target])

        with h5py.File(target_mask_file, 'r') as f:
            target_mask_array = np.array(f['data'][sel_slice_target])[None]
        if self.debug:
            print("Input array min/max/mean", harray.get_minmeanmediammax(input_array), input_array.dtype)
            print("Target array min/max/mean", harray.get_minmeanmediammax(target_array), target_array.dtype)
            print("Input mask array min/max/mean", harray.get_minmeanmediammax(input_mask_array), input_mask_array.dtype)
            print("Target mask array min/max/mean", harray.get_minmeanmediammax(target_mask_array), target_mask_array.dtype)




        # Scale it from uint to float 0..1
        input_array = harray.scale_minmax(input_array * input_mask_array)
        target_array = harray.scale_minmax(target_array * target_mask_array)

        # Everything is already real valued.. so we can use the original resize command.
        if self.transform_resize:
            # This can probably be done prettier.. but yeah..
            resize_shape = self.resize_list[self.resize_index]

            input_array = self.perform_resize(input_array, resize_shape=resize_shape)
            target_array = self.perform_resize(target_array, resize_shape=resize_shape)
            target_mask_array = self.perform_resize(target_mask_array, resize_shape=resize_shape)
            input_mask_array = self.perform_resize(input_mask_array, resize_shape=resize_shape)

        input_tensor = torch.from_numpy(input_array.astype(float)).float()
        target_tensor = torch.from_numpy(target_array.astype(float)).float()
        input_mask_tensor = torch.from_numpy(input_mask_array.astype(float)).float()
        target_mask_tensor = torch.from_numpy(target_mask_array.astype(float)).float()

        container_dict = {'input': input_tensor, 'target': target_tensor,
                          'mask_input': input_mask_tensor, 'mask_target': target_mask_tensor}

        self.debug_display_counter += 1

        # Add more Transformations only when we are training
        # This is used so that we have the same thing on input and target
        # Useful for example when rotating stuff
        random_seed_fixed = np.random.randint(123456789)
        if self.transform_compose is not None and self.dataset_type == 'train':
            for key, value in container_dict.items():
                torch.manual_seed(random_seed_fixed)
                temp = value
                for i_transform in self.transform_compose.transforms:
                    # Only perform Random Erasing on the input
                    # (Or: when we have something unequal to input, go to the next iteration)
                    only_on_input_keys = ["GaussianBlur", "RandomErasing"]
                    transform_name = i_transform._get_name()
                    only_on_input_bool = [x == transform_name for x in only_on_input_keys]
                    if any(only_on_input_bool) and (key != 'input'):
                        continue

                    temp = i_transform(temp)
                container_dict[key] = temp

        return container_dict


if __name__ == "__main__":
    import helper.plot_class as hplotc
    dg_obj = DataGeneratorProstateWeighting(ddata='/local_scratch/sharreve/mri_data/prostate_weighting_h5',
                                            debug=True, optional_target_appendix="_corrected",
                                            switch_input_output=True)

    cont = dg_obj.__getitem__(100)
    plot_obj = hplotc.ListPlot([cont['input'], cont['target']])
    plot_obj.figure.savefig('/local_scratch/sharreve/test.png')
