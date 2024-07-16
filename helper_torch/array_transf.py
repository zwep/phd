# encoding: utf-8


"""

"""

import numbers

import torch
import numpy as np


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def torch_kspace_to_image(x, dim, normalize=True):
    ndim = len(dim)
    y = fftshift(torch.fft(ifftshift(x, dim=dim), signal_ndim=ndim, normalized=normalize), dim=dim)
    return y


def torch_image_to_kspace(y, dim, normalize=True):
    ndim = len(dim)
    x = fftshift(torch.ifft(ifftshift(y, dim=dim), signal_ndim=ndim, normalized=normalize), dim=ndim)
    return x


def scale_minmax(x, dim=(-2, -1)):
    min_x = x
    for i, i_dim in enumerate(dim):
        min_x, _ = min_x.min(dim=i_dim + i)

    max_x = x
    for i, i_dim in enumerate(dim):
        max_x, _ = max_x.max(dim=i_dim + i)

    return (x - min_x)/(max_x - min_x)

def scale_11(x, dim=(-2, -1)):
    min_x = x
    for i, i_dim in enumerate(dim):
        min_x, _ = min_x.min(dim=i_dim, keepdims=True)

    max_x = x
    for i, i_dim in enumerate(dim):
        max_x, _ = max_x.max(dim=i_dim, keepdims=True)


    return 2 * (x - min_x)/(max_x - min_x) - 1


if __name__ == "__main__":
    import os
    import helper.plot_fun as hplotf
    import numpy as np
    import helper.array_transf as harray
    dir_data = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels/train/input'
    file_list = os.listdir(dir_data)
    i_file = os.path.join(dir_data, file_list[0])
    A = np.load(i_file)
    a_sel = A[0][0]
    # hplotf.plot_3d_list(a_sel[np.newaxis], augm='np.abs')

    k_numpy = harray.transform_image_to_kspace_fftn(a_sel)
    # hplotf.plot_3d_list([k_numpy.real, k_numpy.imag], augm='np.real')

    import torch
    import torch.nn

    a_tens = torch.stack([torch.as_tensor(a_sel.real), torch.as_tensor(a_sel.imag)], dim=-1)[np.newaxis].float()

    # Image to kspace
    k = fftshift(torch.fft(ifftshift(a_tens, dim=(-3, -2)), signal_ndim=2, normalized=False), dim=(-3, -2))
    k = k.permute((0, 3, 1, 2))
    conv_layer = torch.nn.Conv2d(in_channels=2, out_channels=2, groups=2, kernel_size=3, padding=1)
    k = conv_layer(k)
    # k = conv_layer(k)
    # k = conv_layer(k)
    # k = conv_layer(k)
    k = k.permute((0, 2, 3, 1))
    img = fftshift(torch.ifft(ifftshift(k, dim=(-3, -2)), signal_ndim=2, normalized=False), dim=(-3, -2))
    img_np = img.detach().numpy()
    z = img_np[:, :, :, 0] + 1j * img_np[:, :, :, 1]
    hplotf.plot_3d_list(z, augm='np.abs')
    hplotf.plot_3d_list(z - a_sel, augm='np.abs')

    x_normal = harray.transform_kspace_to_image_fftn(z)
    hplotf.plot_3d_list(x_normal[np.newaxis], augm='np.abs')

    # Kspace to Image
    img = fftshift(torch.ifft(ifftshift(k, dim=(0, 1)), signal_ndim=2), dim=(0, 1))
    img_cpx = img[:, :, 0].numpy() + 1j * img[:, :, 1].numpy()
    hplotf.plot_3d_list(img.permute((2, 0, 1)))
    hplotf.plot_3d_list(img_cpx[np.newaxis], augm='np.abs')
    hplotf.plot_3d_list(img_cpx[np.newaxis] - a_sel, augm='np.abs')

