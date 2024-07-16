
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy.signal
import scipy.ndimage as scndimage
import scipy.misc
from collections.abc import Sequence

import numpy as np
import torch.nn
from PIL import Image
import helper.plot_class as hplotc

"""
Define a set of transforms over here... if needed...
"""

# https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/src/pre_processing.py

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class TransformElastic:
    def __init__(self, mean=0, std=4, grid_size=(32, 32, 32), mode='reflect', prob=True):
        self.displacement = np.random.normal(mean, std, size=(len(grid_size),) + grid_size)
        self.displacement[-1] = 0  # Setting the displacement in the z-axis to zero.
        self.prob = prob
        self.mode = mode

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        if 'complex' == img.dtype:
            res_real = elasticdeform.deform_grid(img.real, displacement=self.displacement, mode=self.mode)
            res_img = elasticdeform.deform_grid(img.imag, displacement=self.displacement, mode=self.mode)
            res = res_real + 1j * res_img
        else:
            res = elasticdeform.deform_grid(img, displacement=self.displacement, mode=self.mode)

        return res


class TransformRotate:
    def __init__(self, angle_range=(-10, 10), axes=(-3, -2), prob=True):
        self.angle_range = angle_range

        self.axes = axes
        self.prob = prob

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        angle = np.random.randint(*self.angle_range)
        res = scndimage.rotate(img, angle, axes=self.axes, reshape=False)

        return res


class TransformFlip:
    def __init__(self, axes=(-3, -2, -1), prob=True):
        self.axes = axes
        self.prob = prob

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        ax = np.random.choice(self.axes, 1)
        res = np.flip(img, ax)

        return res


class TransformGaussianNoise:
    def __init__(self, mean=0, std=0.1, prob=True):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        res = img + np.random.normal(self.mean, self.std, img.shape)

        return res


class TransformUniformNoise:
    def __init__(self, low=0, high=0.5, prob=True):
        self.low = low
        self.high = high
        self.prob = prob

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        res = img + np.random.uniform(self.low, self.high, img.shape)

        return res


class TransformBrightness:
    def __init__(self, value=0.1, min_value=0, max_value=1, prob=True):
        self.value = value
        self.prob = prob
        self.min_value = min_value
        self.max_value = max_value

    @staticmethod
    def ceil_floor(img, min_value=0, max_value=1):
        img[img > max_value] = max_value
        img[img < min_value] = min_value
        return img

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        random_value = np.random.uniform(0, self.value)
        res = self.ceil_floor(img + random_value, min_value=self.min_value, max_value=self.max_value)

        return res


class TransformStandardize:
    def __init__(self, prob=True):
        self.prob = prob

    def __call__(self, img, ax=None):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        temp_mean = np.mean(img, axis=ax)
        temp_std = np.std(img, axis=ax)

        if isinstance(ax, tuple):
            new_shape = temp_mean.shape + (1,) * len(ax)  # Add new axes...
            temp_mean = temp_mean.reshape(new_shape)
            temp_std = temp_std.reshape(new_shape)
        elif isinstance(ax, int):
            new_shape = temp_mean.shape + (1,)  # Add one more ax
            temp_mean = temp_mean.reshape(new_shape)
            temp_std = temp_std.reshape(new_shape)
        elif ax is None:
            pass  # Dealing with onthing but scalars...
        else:
            print('errr')

        res = (img - temp_mean) / temp_std
        return res


class TransformNormalize:
    def __init__(self, min=0, max=1, prob=True):
        self.min = min
        self.max = max
        self.prob = prob

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        res = (img - np.min(img)) * (self.max - self.min) / (np.max(img) - np.min(img)) + self.min

        return res


class TransformSmooth:
    def __init__(self, kernel_size, mode='same', prob=True):
        self.kernel = np.ones((kernel_size, kernel_size)) / kernel_size
        self.mode = mode
        self.prob = prob

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        res = scipy.signal.convolve2d(img, self.kernel, mode=self.mode)

        return res


# class Resize(torch.nn.Module):
#     """
#     Seb: Modification to the torchvision Resize. Now accepts lists and chooses one by chance.
#     This removes the need to set the random size after the training loop
#
#     Resize the input image to the given size.
#     The image can be a PIL Image or a torch Tensor, in which case it is expected
#     to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
#
#     Args:
#         size (sequence or int): Desired output size. If size is a sequence like
#             (h, w), output size will be matched to this. If size is an int,
#             smaller edge of the image will be matched to this number.
#             i.e, if height > width, then image will be rescaled to
#             (size * height / width, size).
#             In torchscript mode padding as single int is not supported, use a tuple or
#             list of length 1: ``[size, ]``.
#         interpolation (int, optional): Desired interpolation enum defined by `filters`_.
#             Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
#             and ``PIL.Image.BICUBIC`` are supported.
#     """
#
#     def __init__(self, size, interpolation=Image.BILINEAR):
#         super().__init__()
#         if isinstance(size, list):
#             self.size_list = size
#         else:
#             self.size_list = [size]
#
#         self.current_index
#         self.interpolation = interpolation
#
#     def forward(self, img):
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be scaled.
#
#         Returns:
#             PIL Image or Tensor: Rescaled image.
#         """
#         return F.resize(img, self.size, self.interpolation)
#
#     def __repr__(self):
#         interpolate_str = _pil_interpolation_to_str[self.interpolation]
#         return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


if __name__ == "__main__":

    from scipy import misc
    import matplotlib.pyplot as plt
    import helper.plot_fun as hplotf
    b = misc.face(gray=True)/255

    # list_transform = [TransformRotate, TransformBrightness, TransformFlip, TransformGaussianNoise, TransformNormalize, TransformStandardize, TransformUniformNoise, TransformElastic]
    # list_transform = [x() for x in list_transform]
    # # torch_comp = torchvision.transforms.Compose(list_transform)
    #
    # result = torch_comp(b[:,:, np.newaxis])
    # np.random.binomial(len(torch_comp.transforms), 0.5, len(torch_comp.transforms))
    #
    # random_sel = np.random.choice(torch_comp.transforms, size=2, replace=False)
    # for i_trans in torch_comp.transforms:
    #     plt.figure()
    #     new_comp = torchvision.transforms.Compose([i_trans])
    #     result = new_comp(b[:, :, np.newaxis])
    #     plt.imshow(result[:, :, 0])
    #
    # torch_comp(b)
    # original = np.array(b)

    import numpy, imageio, elasticdeform

    displacement = np.random.normal(50, 30, size=(2,) + (2, 3))
    res = elasticdeform.deform_grid(b[np.newaxis], displacement=displacement, axis=(1, 2))
    plt.imshow(res[0])


    # Testing image trasnforms...

    import data_generator.UnetValidation

    ddata = "/home/bugger/Documents/data/grand_challenge/data"
    input_shape = (136, 136, 122)
    a_gen = data_generator.UnetValidation.UnetValidation(ddata=ddata, input_shape=input_shape)
    x0, y0 = a_gen.__getitem__(0)
    mean = 0
    std = 4
    grid_size = (32, 32)
    displacement = np.random.normal(mean, std, size=(2,) + grid_size)
    plt.quiver(displacement[0], displacement[1])
    plt.imshow(displacement[0])

    input_shape = tuple(x0.shape[1:])
    x_dim, y_dim, z_dim = input_shape
    dz = gaussian_filter(np.random.randn(*input_shape), 3, mode="constant", cval=0) * 15
    dy = gaussian_filter(np.random.randn(*input_shape), 3, mode="constant", cval=0) * 15
    dx = gaussian_filter(np.random.randn(*input_shape), 3, mode="constant", cval=0) * 15
    y, x, z = np.meshgrid(np.arange(y_dim), np.arange(x_dim), np.arange(z_dim), indexing='ij')
    indices = y + dy, x + dx, z + dz
    derp = map_coordinates(x0[0], indices, order=3, mode='reflect')
    hplotc.SlidingPlot(derp)
    hplotc.SlidingPlot(x0[0])

    t_standard = TransformStandardize()
    x0 = t_standard(x0.numpy())
    x_def = elasticdeform.deform_grid(x0[0], displacement=displacement, axis=(1, 2))
    import helper.plot_class as hplotc

    hplotc.SlidingPlot(x_def)
    hplotc.SlidingPlot(x0)
