# encoding: utf-8

import re
import random
import torch.utils.data
import numpy as np
import os
import helper.array_transf as harray
import data_generator.Generic as generic_data_generator
import torch.utils.data
import collections

"""
Here we have the data generators for the Survey 2 B1 maps

either single channel.. or all channels..
"""

import data_generator.Generic as data_gen
import nibabel as nib
import helper_torch.transforms as htransform


class UnetValidation(data_gen.DatasetGeneric):
    def __init__(self, ddata, input_shape, target_shape=None, batch_perc=0.010, transform=None,
                 shuffle=True, dataset_type='train', file_ext='nii.gz', **kwargs):

        input_args = {k: v for k, v in locals().items() if (k !='self') and (k !='__class__')}
        super().__init__(**input_args)

        self.correct_elas_pixels = 10

        self.width = input_shape[0] + self.correct_elas_pixels  # This is the max...
        self.depth = input_shape[-1] + self.correct_elas_pixels  # This is the max as well... we can go lower..

        if target_shape is None:
            self.width_target = 44 + self.correct_elas_pixels  # This is different from input.. because of unpadded conv.
            self.depth_target = 28 + self.correct_elas_pixels  # This is the max as well... we can go lower..
        else:
            self.width_target = target_shape[0] + self.correct_elas_pixels
            self.depth_target = target_shape[-1] + self.correct_elas_pixels

        self.shift_center = np.array([self.width, self.width, self.depth])

    @staticmethod
    def _get_range(center, delta_x):
        temp_range = np.arange(center - delta_x // 2, center + delta_x // 2)
        return temp_range

    def __getitem__(self, index, x_center=None, y_center=None, t_center=None):
        """Generate one batch of data"""

        if self.debug:
            print('UnetValidation Generator __getitem__')

        # Initialize the transforms....
        norm_trans = htransform.TransformNormalize(prob=False)  # Only X
        elas_trans = htransform.TransformElastic(mean=0, std=0.4, mode='constant')  # Part of the previous exercise
        unif_noise_trans = htransform.TransformUniformNoise(low=0, high=0.1)  # Only X
        bright_trans = htransform.TransformBrightness(value=0.05)  # Only X
        flip_trans = htransform.TransformFlip()

        if self.debug:
            print('\t Transformers created')

        i_file = self.file_list[index]
        input_file = os.path.join(self.input_dir, i_file)
        if self.input_is_output:
            target_file = os.path.join(self.target_dir, i_file)
        else:
            target_file = os.path.join(self.target_dir, re.sub('image', 'mask', i_file))

        # Will be of shape 8, 8, X, Y
        x = nib.load(input_file).get_fdata()
        y = nib.load(target_file).get_fdata()

        # Normalize based on CT values higher than 1000 are not interesting...
        x = x / 1000
        x[x > 1] = 1

        if self.debug:
            print('EXEC - GEN: Loaded all data')

        n_x, n_y, n_z = x.shape
        if self.debug:
            print('\t loaded image shape ', x.shape)
            print('\t loaded target image shape ', y.shape)
            print('\t With index ', index, input_file, target_file)

        # # # For loop on data...
        x_center = np.random.randint(self.width // 2, n_x - self.width // 2 + 1)
        y_center = np.random.randint(self.width // 2, n_y - self.width // 2 + 1)
        t_center = np.random.randint(self.depth // 2, n_z - self.depth // 2 + 1)

        x_range_target = self._get_range(x_center, self.width_target)
        y_range_target = self._get_range(y_center, self.width_target)
        t_range_target = self._get_range(t_center, self.depth_target)

        # num_target = 0
        # while num_target == 0 and np.random.uniform(0, 1) > 0.05:
        #     x_center = np.random.randint(self.width // 2, n_x - self.width // 2 + 1)
        #     y_center = np.random.randint(self.width // 2, n_y - self.width // 2 + 1)
        #     t_center = np.random.randint(self.depth // 2, n_z - self.depth // 2 + 1)
        #
        #     x_range_target = self._get_range(x_center, self.width_target)
        #     y_range_target = self._get_range(y_center, self.width_target)
        #     t_range_target = self._get_range(t_center, self.depth_target)
        #
        #     y_check = np.take(y, x_range_target, axis=-3)
        #     y_check = np.take(y_check, y_range_target, axis=-2)
        #     y_check = np.take(y_check, t_range_target, axis=-1)
        #     num_target = y_check.sum()

        x_range = self._get_range(x_center, self.width)  # Extra pixel size to correct for elas deform
        y_range = self._get_range(y_center, self.width)
        t_range = self._get_range(t_center, self.depth)
        
#        if self.debug:
 #           print(f'\t Using x/y/z range {x_range} {y_range} {t_range}')
            # print(f'\t Using x/y/z target range{x_range_target}, {y_range_target}, {t_range_target}')

        x = np.take(x, x_range, axis=-3)
        x = np.take(x, y_range, axis=-2)
        x = np.take(x, t_range, axis=-1)

        y = np.take(y, x_range_target, axis=-3)
        y = np.take(y, y_range_target, axis=-2)
        y = np.take(y, t_range_target, axis=-1)

        if self.debug:
            print('\t  subsampled image')
            print(f'\t \t  shape x/y {x.shape}/{y.shape}')
            print('\t \t  counter unique values target', collections.Counter(y.ravel()))

        # Only perform these data operations during training...
        if self.dataset_type == 'train':
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            np.random.seed(seed)  # Used so that the probs are all executed the same
            x = elas_trans(x)  # Part of the previous exercise
            x = flip_trans(x)
            x = unif_noise_trans(x)
            x = bright_trans(x)
            x = norm_trans(x)

            np.random.seed(seed)  # Used so that the probs are all executed the same
            y = elas_trans(y)  # Part of the previous exercise
            y = flip_trans(y)
            y = np.round(y)

        # y[y > 2] = 2 # Part of the previous exercise
        y[y > 1] = 1  # This is one of the drastic changes..
        y[y < 0] = 0  # I hope this is the answer

        if self.debug:
            print('\t Transformations applied')
            print(f'\t \t  shape x/y {x.shape}/{y.shape}')

        # Undo the correction for the elastic transform thingy
        x = x[self.correct_elas_pixels//2:-self.correct_elas_pixels//2,
              self.correct_elas_pixels//2:-self.correct_elas_pixels//2,
              self.correct_elas_pixels//2:-self.correct_elas_pixels//2]

        y = y[self.correct_elas_pixels//2:-self.correct_elas_pixels//2,
              self.correct_elas_pixels//2:-self.correct_elas_pixels//2,
              self.correct_elas_pixels//2:-self.correct_elas_pixels//2]

        # x = torch.cat(3 * [torch.as_tensor(x[np.newaxis])]) # Part of the previous exercise
        x = torch.from_numpy(x[np.newaxis].copy())
        y = torch.from_numpy(y[np.newaxis].copy())

        if self.debug:
            print('\t returning tensors')
            print('\t  \t  counter unique values target', collections.Counter(y.numpy().ravel()))

        return x.float(), y.float()


if __name__ == "__main__":

    import helper.plot_class as hplotc
    import data_generator.UnetValidation as data_gen
    import torch
    import torch.nn.functional as F
    import torch.nn
    import helper_torch.loss as htloss
    import numpy as np
    import os
    import importlib
    import re
    importlib.reload(data_gen)
    dir_data = '/home/bugger/Documents/data/grand_challenge/data'
    # dir_data = '/home/seb/data/grand_challenge'
    get_shape = (430, 430, 64)
    # get_shape = (420, 420, 122)
    res = data_gen.UnetValidation(ddata=dir_data, input_shape=get_shape, target_shape=get_shape,
                                  dataset_type='train',
                                  debug=True, shuffle=False)
    a, b = res.__getitem__(0)
    hplotc.SlidingPlot(np.moveaxis(a[0].numpy(), -1, 0))

    import helper_torch.loss as hloss
    lossobj = hloss.DiceLoss()

    hplotc.SlidingPlot(np.moveaxis(a[0].numpy(), -1, 0))
    hplotc.SlidingPlot(np.moveaxis(b[0].numpy(), -1, 0))
    a = nib.load(os.path.join(res.input_dir, res.file_list[0]))
    b = nib.load(os.path.join(res.target_dir, re.sub('image', 'mask', res.file_list[0])))
    a = a.get_fdata()
    b = b.get_fdata()

    import helper_torch.transforms as htransform
    elas_trans = htransform.TransformElastic(mean=0, std=4, prob=False, mode='constant')
    a_trans = elas_trans(a)
    b_trans = elas_trans(b)
    b_trans_sub = elas_trans(b[:, :, 300:400])
    b_trans_sub_subsub = elas_trans(b[140:170, 200:230, 300:400])
    hplotc.SlidingPlot(a)
    hplotc.SlidingPlot(np.moveaxis(a_trans, -1, 0))
    hplotc.SlidingPlot(np.moveaxis(b_trans, -1, 0))
    hplotc.SlidingPlot(np.moveaxis(b_trans_sub, -1, 0))
    hplotc.SlidingPlot(np.moveaxis(b_trans_sub_subsub, -1, 0))

    hplotc.SlidingPlot(np.moveaxis(b, -1, 0))

    # res = data_gen.UnetValidation(ddata=dir_data, input_shape=get_shape, debug=True, shuffle=False)
    res_A, res_B = res.__getitem__(0, 430//2, 430//2, 730//2)

    x = res_A.numpy()[0]

    y = res_B.numpy()
    import random
    import helper_torch.transforms as htransform
    importlib.reload(htransform)

    seed = np.random.randint(2147483647)  # make a seed with numpy generator
    random.seed(seed)  # apply this seed to img transforms
    norm_trans = htransform.TransformNormalize()  # Only X

    x = norm_trans(x)
    unif_noise_trans = htransform.TransformUniformNoise(prob=False)  # Only X
    x = unif_noise_trans(x)
    hplotc.SlidingPlot(np.moveaxis(x, -1, 0))
    hplotc.SlidingPlot(np.moveaxis(y, -1, 0))

    bright_trans = htransform.TransformBrightness(prob=False)  # Only X
    x = bright_trans(x)
    hplotc.SlidingPlot(x)

    rot_trans = htransform.TransformRotate(angle_range=(-5, 5), prob=False, axes=(-2, -1 ))  # This is not so nice....
    x = rot_trans(x)
    hplotc.SlidingPlot(x)
    hplotc.SlidingPlot(np.moveaxis(x, -1, 0))

    flip_trans = htransform.TransformFlip(prob=False)
    x = flip_trans(x)
    hplotc.SlidingPlot(x)

    elas_trans = htransform.TransformElastic(mean=0, std=4, prob=False, mode='constant')
    # hplotc.SlidingPlot(elas_trans.displacement[:, :, :, 0], ax_3d=True)
    x = res_A.numpy()[0]
    x1 = elas_trans(x)
    x1 = norm_trans(x1)
    hplotc.SlidingPlot(np.moveaxis(x1, -1, 0))
    hplotc.SlidingPlot(x1)

    import numpy as np
    import helper_torch.transforms as htransform
    import helper.plot_class as hplotc
    y = np.random.rand(136, 136)
    n = 30
    y[y.shape[0] // 2 - n: y.shape[0] // 2 + n, y.shape[1] // 2 - n: y.shape[1] // 2 + n] = 2
    n = 15
    y[y.shape[0] // 2 - n: y.shape[0] // 2 + n, y.shape[1] // 2 - n: y.shape[1] // 2 + n] = 1

    elas_trans = htransform.TransformElastic(mean=0, std=4, prob=False, mode='mirror', grid_size=(8, 8))
    y1 = elas_trans(y.astype(np.float))
    y1 = np.round(y1)
    y1[y1 > 2] = 2

    y1_shift = np.zeros((3,) + y1.shape)
    y1_shift[1] = 1 * (y1 == 1)
    y1_shift[2] = 2 * (y1 == 2)

    hplotc.SlidingPlot(y1_shift)
    hplotc.SlidingPlot(y1)
    hplotc.SlidingPlot(y)

    y = torch.as_tensor(y)
    y1_shift = torch.as_tensor(y1_shift)

    import helper_torch.loss as hloss
    import torch.nn
    import importlib
    importlib.reload(hloss)
    wel = hloss.WeightedCrossEntropyLoss()
    el = torch.nn.CrossEntropyLoss()
    y1_shift[2] = y1_shift[2] + np.random.normal(0, 1, size=(y1_shift[2].shape))
#    y[0][0] = np.inf
 #   y1_shift[0][0][0] = np.inf
    wel(torch.as_tensor(y1_shift[np.newaxis]), torch.as_tensor(y[np.newaxis]).long())
    wel(y1_shift[np.newaxis], y[np.newaxis])
    hplotc.SlidingPlot(np.moveaxis(y, -1, 0))



    # testing misc
    import nibabel as nib
    input_file = '/home/bugger/Documents/data/acdc/training/patient001/patient001_4d.nii.gz'
    x = nib.load(input_file)
    x = x.get_fdata().T
    hplotc.SlidingPlot(x)

    elas_trans = htransform.TransformElastic(mean=0, std=4, mode='constant')

    import scipy.io
    input_file = '/home/bugger/Documents/data/york/mrimages/sol_yxzt_pat1.mat'
    x = scipy.io.loadmat(input_file)
    x.keys()
    x = x['sol_yxzt'].T
    hplotc.SlidingPlot(np.moveaxis(x, 1, -1))

