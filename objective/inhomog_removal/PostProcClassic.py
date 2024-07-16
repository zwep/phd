import os
import numpy as np
import h5py
import nibabel

import helper.array_transf as harray
import helper.misc as hmisc
from skimage.util import img_as_ubyte, img_as_int, img_as_uint
from biasfield_algorithms.N4ITK import get_n4itk


class PostProcClassic:
    def __init__(self, img_dir, mask_dir, dest_dir, **kwargs):
        self.image_dir = img_dir
        self.mask_dir = mask_dir
        self.target_dir = dest_dir
        self.storage_extension = kwargs.get('storage_extension', 'npy')
        self.mask_suffix = kwargs.get('mask_suffix', '')
        # In case it doesnt exist yet... make it
        if not os.path.isdir(self.target_dir):
            os.makedirs(self.target_dir)

        self.file_list = os.listdir(img_dir)
        self.sum_of_absolute_img = None
        self.loaded_image = None
        self.loaded_mask = None
        self.n_slices = None

    def run(self):
        # Do a complete run...fixed to to a patch-based evaluation
        for i_index, i_file in enumerate(self.file_list):
            print('Running file ', i_file)
            self.load_file(file_index=i_index)
            # This contains the corrected, biasfield and multi-coil corrected
            file_result = self.run_loaded_file()
            corrected_result = np.array([x['corrected'] for x in file_result])
            self.save_array(corrected_result, i_file, self.storage_extension)
            biasfield = np.array([x['biasfield'] for x in file_result])
            self.save_array(biasfield, 'biasfield_' + i_file, self.storage_extension)
            self.save_array(self.sum_of_absolute_img, 'uncorrected_' + i_file, self.storage_extension)

    def save_array(self, x_img, file_name, extension):
        file_name, _ = os.path.splitext(file_name)
        target_file = os.path.join(self.target_dir, file_name)
        x_img = harray.treshold_percentile(x_img, q=98)
        x_img = harray.scale_minmax(x_img)
        # Saves some data....
        x_img = img_as_ubyte(x_img)
        if 'npy' in extension:
            np.save(x_img, target_file + '.npy')
        elif 'nii' in extension:
            x_img = x_img.T[::-1, ::-1]
            nibabel_result = nibabel.Nifti1Image(x_img, np.eye(4))
            nibabel.save(nibabel_result, target_file + '.nii.gz')
        elif 'h5' in extension:
            with h5py.File(target_file + '.h5', 'w') as f:
                f.create_dataset('data', data=x_img)
        else:
            print("Unknown data extension: ", extension)
            print("Please use npy or h5")

    def load_file(self, file_index):
        # Twijfel een beetje aan de file counter...
        sel_file = self.file_list[file_index]
        sel_img = os.path.join(self.image_dir, sel_file)
        # Is this always the case? Dont I need some appendix sometime for the mask files?
        sel_file_name, sel_file_ext = os.path.splitext(sel_file)
        sel_mask = os.path.join(self.mask_dir, sel_file_name + self.mask_suffix + sel_file_ext)

        # Stuff...
        self.loaded_image = hmisc.load_array(sel_img)
        self.loaded_mask = hmisc.load_array(sel_mask)
        if self.loaded_mask.ndim == 2:
            self.loaded_mask = self.loaded_mask[None]
        print("Loaded image shape ", self.loaded_image.shape)
        print("Loaded mask shape ", self.loaded_mask.shape)
        n_chan_or_n_slice = self.loaded_image.shape[0]
        if n_chan_or_n_slice == 8:
            self.n_chan = 8
            self.n_slices = 1
            # Make the img size (n_loc=1, nx ny)
            self.sum_of_absolute_img = np.abs(self.loaded_image).mean(axis=0)[None]
        else:
            self.n_chan = 1
            self.n_slices = n_chan_or_n_slice
            self.sum_of_absolute_img = np.abs(self.loaded_image)

        self.sum_of_absolute_img = harray.scale_minmax(self.sum_of_absolute_img)

    def run_loaded_file(self):
        file_result = []
        for i_slice in range(self.n_slices):
            print(f'Running slice {i_slice} / {self.n_slices}', end='\r')
            sel_slice = self.sum_of_absolute_img[i_slice]
            sel_mask = self.loaded_mask[i_slice]
            output, output_n4itk = get_n4itk(sel_slice, mask=sel_mask, n_fit=4, n_iterations=100, output_biasfield=True)
            temp = {'corrected': output, 'biasfield': output_n4itk}
            file_result.append(temp)
        return file_result
