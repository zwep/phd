import os
import numpy as np
import skimage.transform
import helper.misc as hmisc
import helper.array_transf as harray
from skimage.util import img_as_ubyte


class FileGather:
    def __init__(self, data_dir, mask_dir=None):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.file_list = os.listdir(self.data_dir)
        self.img_list = self.load_file_list()

    def print_img_shape(self):
        for i_file in self.file_list:
            file_path = os.path.join(self.data_dir, i_file)
            loaded_array = hmisc.load_array(file_path)
            print(i_file, loaded_array.shape, loaded_array.dtype)

    @staticmethod
    def prep_image(x, file_name):
        if 'nii.gz' in file_name:
            x = x.T[:, ::-1, ::-1]

        if x.ndim == 3:
            n_slice = x.shape[0]
            # Could also test for complex dtype
            if n_slice == 8:
                # In this case we have 8 coils
                x = np.abs(x).sum(axis=0)
            else:
                x = np.abs(x[n_slice // 2])

        x = img_as_ubyte(harray.scale_minmax(x))
        x = skimage.transform.resize(x, (256, 256))
        return x

    def load_file_list(self):
        img_list = []
        for i_file in self.file_list:
            file_path = os.path.join(self.data_dir, i_file)
            loaded_array = hmisc.load_array(file_path)
            loaded_array = self.prep_image(loaded_array, i_file)
            if self.mask_dir is not None:
                mask_path = os.path.join(self.mask_dir, i_file)
                loaded_mask = hmisc.load_array(mask_path).astype(int)
                loaded_mask = self.prep_image(loaded_mask, i_file)
                loaded_array = loaded_array * loaded_mask

            img_list.append(loaded_array)

        return img_list