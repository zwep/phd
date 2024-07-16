import torch.utils.data
import helper.plot_class as hplotc
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

"""
Data gen for reconstruction

Borrowed/copied from the code I made for DIRECT
"""


def binary_discretize_spokes(coordinates, shape):
    # This gives a surrogate mask that can be used..?
    nx, ny = shape
    discretize_spokes = np.zeros((nx, ny))
    delta_range = range(-1, 2)
    for i_coord in coordinates.astype(int):
        ix, iy = i_coord
        # Also add some more room...
        for delta_space_x in delta_range:
            for delta_space_y in delta_range:
                floor_x = int(np.floor(ix + delta_space_x + nx/2))
                ceil_x = int(np.ceil(ix + delta_space_x + nx/2))
                floor_y = int(np.floor(iy + delta_space_y + ny / 2))
                ceil_y = int(np.ceil(iy + delta_space_y + ny / 2))
                # Make sure that we discretize more...
                discretize_spokes[min(nx - 1, floor_x), min(ny - 1, floor_y)] = 1
                discretize_spokes[min(nx - 1, floor_x), min(ny - 1, ceil_y)] = 1
                discretize_spokes[min(nx - 1, ceil_x), min(ny - 1, floor_y)] = 1
                discretize_spokes[min(nx - 1, ceil_x), min(ny - 1, ceil_y)] = 1
    return discretize_spokes


class DataGeneratorReconstruction(data_gen.DatasetGenericComplex):
    def __init__(self, ddata, input_shape=None, target_shape=None,
                 shuffle=True, dataset_type='train', file_ext='npy', transform=None, **kwargs):
        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, transform=transform, **kwargs)
        # SH: Adding fixed trajectory creation for images of size 256
        self.ovs = 1.25
        self.width = 4
        N = 256
        max_spokes = int(np.ceil((np.pi / 2) * N))
        n_points = N
        img_size = (N, N)
        self.trajectory_radial = sigpy.mri.radial(coord_shape=(max_spokes, n_points, 2), img_shape=img_size, golden=False)

    def undersample_image(self, img_array, acc=None):
        # SH: I know this is hardcoded and a bad way of programming
        # But Im trying to keep speed
        # acc_factor = np.random.choice([5, 10])
        #if '5x' in self.text_description:
        #    acc_factor = 5
        #else:
        if acc is None:
            acc_factor = np.random.choice([5, 10], 1)[0]
        else:
            acc_factor = acc

        # logger.info(f'! ! ! ! ! Acceleration factor {acc_factor}! ! ! ! ! ')
        spoke_ind = np.random.choice(list(range(acc_factor)))
        # I guess that each acc factor has multiple ways of looking at them..
        # With this I can vary it a little bit
        trajectory = self.trajectory_radial[spoke_ind::acc_factor]
        # logger.info(f'! ! ! ! ! Length spokes {trajectory.shape}! ! ! ! ! ')
        dcf = np.sqrt(trajectory[..., 0] ** 2 + trajectory[..., 1] ** 2)
        input_array = []
        img_shape = img_array.shape[-2:]
        # img_array = img_array.numpy()
        img_array = np.fft.ifft2(np.fft.fftshift(img_array, axes=(-2, -1)), norm='ortho')
        for i_coil in img_array:
            temp_kspace = sigpy.nufft(i_coil, coord=trajectory, width=self.width, oversamp=self.ovs)
            temp_img = sigpy.nufft_adjoint(temp_kspace * dcf, coord=trajectory, oshape=img_shape,
                                           width=self.width, oversamp=self.ovs)
            input_array.append(temp_img)
        input_array = np.array(input_array)
        mask_array = binary_discretize_spokes(trajectory.reshape(-1, 2), shape=img_shape)
        # I could also use pykbnufft.. That handles the Tensors nicely...
        return input_array, mask_array

    def _debug_plot(self, index):
        # Check the index
        container = self.__getitem__(index)
        input_cpx = container['input'][..., 0] + 1j * container['input'][..., 1]
        input_sos = np.sqrt(np.sum(np.abs(np.fft.fft2(input_cpx)) ** 2, axis=0))
        fig_obj = hplotc.ListPlot([input_cpx, input_sos, container['mask'], container['target']], cbar=True)
        fig_obj.figure.savefig(os.path.expanduser(f'~/test_reconstruction_{index}.png'))
    def __getitem__(self, index):
        """Generate one batch of data"""
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        # Select a file
        i_file = file_list[index]
        # Creating input File object
        input_file = os.path.join(input_dir, i_file)
        with h5py.File(input_file, 'r') as f:
            n_card = f['kspace'].shape[0]
        sel_card = np.random.choice(range(n_card), 1)[0]
        fs_kspace = hmisc.load_array(input_file, data_key='kspace', sel_slice=sel_card)
        fs_kspace_cpx = fs_kspace[..., ::2] + 1j * fs_kspace[..., 1::2]  # Convert real-valued to complex-valued data.
        fs_kspace_cpx = np.ascontiguousarray(fs_kspace_cpx.transpose(2, 0, 1))[-8:]
        # fs_kspace_cpx = harray.scale_minmax(fs_kspace_cpx, is_complex=True)
        # Kspace in... we get img space out
        temp_img_space, temp_mask_array = self.undersample_image(fs_kspace_cpx)
        # Therefore. convert it back to kspace
        cpx_tensor = np.fft.fftshift(np.fft.fft2(temp_img_space, norm='ortho'), axes=(-2, -1))
        # Scale it to -1.. 1
        cpx_tensor = torch.from_numpy(harray.scale_minmax(cpx_tensor, is_complex=True))
        cpx_tensor = np.fft.fft2(cpx_tensor)
        # Create target array
        target_array = np.sqrt(np.sum(np.abs(np.fft.fft2(fs_kspace_cpx)) ** 2, axis=0, keepdims=True))
        target_array = harray.scale_minmax(target_array)
        # Create tensors..
        # input_tensor = torch.view_as_real(cpx_tensor).to(torch.float32)
        input_array = self.transform_complex(cpx_tensor, stack_ax=self.stack_ax)
        input_tensor = torch.from_numpy(input_array).to(torch.float32)
        mask_tensor = torch.from_numpy(temp_mask_array).to(torch.bool)
        target_tensor = torch.as_tensor(target_array).float()
        container_dict = {'input': input_tensor, 'target': target_tensor, 'mask': mask_tensor, 'target_cpx_coil': np.fft.fft2(fs_kspace_cpx)}
        return container_dict


if __name__ == "__main__":
    # from data_generator.Reconstruction import DataGeneratorReconstruction
    ddata = '/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed'
    data_obj = DataGeneratorReconstruction(ddata=ddata, file_ext='h5')
    data_obj._debug_plot(0)