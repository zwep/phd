
import helper.plot_class as hplotc
import torch
import os
import re
import helper.array_transf as harray
import numpy as np
import data_generator.Generic as data_gen
import nibabel
import skimage.transform as sktransf
import biasfield_algorithms.N4ITK as get_n4itk

# # # Quick test...
import helper.misc as hmisc
ddata = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/input_abs_sum/M21_to_5_MR_20210329_0002_transversal.nii.gz'
A = hmisc.load_array(ddata).T[:, ::-1, ::-1]
print('Shape of laoded array ', A.shape)
A_sel = A[0]
import time
t0 = time.time()
get_n4itk.get_n4itk(A_sel, mask=np.ones(A_sel.shape))
print(time.time() - t0)


class DataGeneratorMMSegment(data_gen.DatasetGenericComplex):
    """
    Simple child class which only implements the get_item function

    Idea is that more child-classes can be created for different datasets while maintaining the options easily
    """
    n_classes = 3

    def __getitem__(self, index):
        """Generate one batch of data"""
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']

        index = index % len(file_list)
        i_file = file_list[index]
        file_name_nii, ext_gz = os.path.splitext(i_file)
        file_name, ext_nii = os.path.splitext(file_name_nii)

        input_file = os.path.join(input_dir, i_file)
        target_file = os.path.join(target_dir, file_name + '_gt' + ext_nii + ext_gz)

        input_array = nibabel.load(input_file).get_fdata()
        input_array_shape = input_array.shape
        n_slice = input_array_shape[-1]
        xy_shape = input_array_shape[:2]
        sel_slice = np.random.randint(0, n_slice)
        input_array = input_array[:, :, sel_slice]
        input_array = harray.scale_minmax(input_array)

        input_array = sktransf.resize(input_array, (256, 256), preserve_range=True)
        input_array, _ = get_n4itk.get_n4itk(input_array, n_iterations=25)

        if os.path.isfile(target_file):
            temp_array = nibabel.load(target_file).get_fdata()
            temp_array = temp_array[:, :, sel_slice]
            classes = set(temp_array.ravel())

            target_array = np.zeros((self.n_classes,) + xy_shape)
            for i_class in range(1, self.n_classes+1):
                target_array[i_class-1] = temp_array == i_class

            target_array = sktransf.resize(target_array, (self.n_classes, 256, 256), anti_aliasing=True)
            target_array = (target_array > 0.5).astype(int)
        else:
            if self.debug:
                print('Found file')

            temp_array = target_array = np.zeros((self.n_classes, ) + input_array.shape)

        input_tensor = torch.from_numpy(input_array[None]).float()
        target_tensor = torch.from_numpy(target_array).float() #.long() #float()

        if self.debug:
            container_dict = {'input': input_tensor, 'target': target_tensor, 'temp': temp_array}
        else:
            container_dict = {'input': input_tensor, 'target': target_tensor}

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
                    # (Or: when we have something unequal to input, continue)
                    if i_transform._get_name() == 'RandomErasing' and key != 'input':
                        continue

                    temp = i_transform(temp)

                container_dict[key] = temp


        return container_dict


if __name__ == "__main__":
    dg_obj = DataGeneratorMMSegment(input_shape=(0,0), ddata='/media/bugger/MyBook/data/m&m/MnM_dataset', dataset_type='train',
                                    file_ext='gz', shuffle=False)
    cont = dg_obj.__getitem__(2)
    hplotc.ListPlot([cont['input'], (cont['target']), cont['temp']])

    # Why no proper target..?
    # temp_array = cont['temp']
    # classes = set(temp_array.ravel())
    # n_classes = max(len(classes)-1, 1)
    #
    # target_array = np.zeros((n_classes,) + temp_array.shape)
    # for i_class in range(1, n_classes+1):
    #     target_array[i_class-1] = temp_array == i_class
    #
    # hplotc.ListPlot(target_array)
    # # target_array = sktransf.resize(target_array, (n_classes, 256, 256), anti_aliasing=True, preserve_range=True)
    # target_array = sktransf.resize(target_array, (n_classes, 256, 256), anti_aliasing=True)
    # target_array = (target_array > 0.5).astype(int)
    # hplotc.ListPlot(target_array)
