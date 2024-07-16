# encoding: utf-8

from torch.autograd import Variable
import matplotlib.pyplot as plt
import importlib
from model.VGG import Vgg16
from helper_torch.misc import flatten, extract_patches
import orthopoly
import torch
import torch.nn as nn
import helper_torch.layers as hlayer
import numpy as np
import scipy.special
import torch.nn.functional as F
from typing import Dict, List, cast
from torch import Tensor, einsum
import helper.misc as hmisc
from functools import reduce
from operator import mul
import math
import sklearn.utils.extmath
import collections
from typing import Optional
import warnings

"""
Loss Functions
"""


def cosine_similarity(x, patches, stride, epsilon=10e-10):
    """
    Arguments:
        x: a float tensor with shape [C, H, W].
        patches: a float tensor with shape [M, C, size, size], normalized.
        stride: an integer.
    Returns:
        a float tensor with shape [N, M],
        where N = n * m, n = 1 + floor((H - size)/stride),
        and m = 1 + floor((W - size)/stride).
    """
    M = patches.size(0)
    x = x.unsqueeze(0)
    products = F.conv2d(x, patches, stride=stride)  # shape [1, M, n, m]

    size = patches.size(2)
    x_norms = F.lp_pool2d(x, norm_type=2, kernel_size=size, stride=stride)  # shape [1, C, n, m]
    x_norms = x_norms.norm(p=2, dim=1, keepdim=True)  # shape [1, 1, n, m]
    products /= (x_norms + epsilon)

    return products.squeeze(0).view(M, -1).t()


def squared_l2_distance(x, patches, stride):
    """
    Arguments:
        x: a float tensor with shape [C, H, W].
        patches: a float tensor with shape [M, C, size, size], unnormalized.
        stride: an integer.
    Returns:
        a float tensor with shape [N, M],
        where N = n * m, n = 1 + floor((H - size)/stride),
        and m = 1 + floor((W - size)/stride).
    """

    # compute squared norms of patches
    M = patches.size(0)
    patch_norms = torch.pow(patches, 2).sum(dim=[1, 2, 3])  # shape [M]

    # compute scalar products
    x = x.unsqueeze(0)
    products = F.conv2d(x, patches, stride=stride)  # shape [1, M, n, m]
    n, m = products.size()[2:]
    N = n * m
    products = products.squeeze(0).view(M, N)

    # compute squared norms of patches from x
    size = patches.size(2)
    x_norms = F.lp_pool2d(x, norm_type=2, kernel_size=size, stride=stride)  # shape [1, C, n, m]
    x_norms = torch.pow(x_norms, 2).sum(dim=1).squeeze(0).view(N)  # shape [N]

    # |x - y|^2 = |x|^2 + |y|^2 - 2*(x, y)
    distances = patch_norms.unsqueeze(1) + x_norms.unsqueeze(0) - 2.0 * products  # shape [M, N]
    return distances.t()


"""
Loss Classes
"""


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, **kwargs):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.smoothed_target = kwargs.get('smoothed_target', False)
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_truth):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_truth (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        # Slight difference between CycleGAN and GAN implemetnation stuff..
        if isinstance(target_is_truth, bool):
            check_something = target_is_truth
        else:
            check_something = torch.any(target_is_truth.type(torch.bool))

        if check_something:
            if self.smoothed_target:
                target_tensor = self.real_label - np.random.uniform(0, 0.1)
            else:
                target_tensor = self.real_label
        else:
            if self.smoothed_target:
                target_tensor = self.fake_label + np.random.uniform(0, 0.1)
            else:
                target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_truth):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_truth)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_truth:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class InternalComplexLoss(torch.nn.Module):
    # We assume that the complex parts are in the channel axis
    # Index 0 should contain the real part
    # Index 1 should contain the imaginary part

    def __init__(self, debug=False):
        super().__init__()
        self.loss_obj_abs = torch.nn.L1Loss()
        self.loss_obj_angle = torch.nn.L1Loss()
        self.loss_obj_real = torch.nn.L1Loss()
        self.loss_obj_imag = torch.nn.L1Loss()
        self.debug = debug

    def _get_angle(self, x):
        return torch.atan2(x[:, 1], x[:, 0])

    def _get_abs(self, x):
        return torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)

    def forward(self, x_pred, x_true):
        x_pred_split = x_pred.split(2, dim=1)
        x_true_split = x_true.split(2, dim=1)
        # print('\n ComplexLoss length split: ', len(x_pred_split))
        loss_abs = 0
        loss_angle = 0
        loss_real = 0
        loss_imag = 0
        for i_pred, i_true in zip(x_pred_split, x_true_split):
            # temp_loss_abs = self.loss_obj_abs(self._get_abs(i_pred), self._get_abs(i_true))
            # temp_loss_angle = self.loss_obj_angle(self._get_angle(i_pred), self._get_angle(i_true))
            temp_loss_real = self.loss_obj_real(i_pred[:, 0], i_pred[:, 0]) - self.loss_obj_real(i_true[:, 0], i_true[:, 0])
            temp_loss_imag = self.loss_obj_real(i_pred[:, 1], i_pred[:, 1]) - self.loss_obj_real(i_true[:, 1], i_true[:, 1])
            # loss_abs += temp_loss_abs
            # loss_angle += temp_loss_angle
            loss_real += temp_loss_real
            loss_imag += temp_loss_imag

        return loss_real + loss_imag + loss_angle + loss_abs


class InputDistanceLoss(torch.nn.Module):
    # Lol.. checking if we can optimize relative to the input..
    def __init__(self):
        super().__init__()
        self.loss_obj_input = torch.nn.L1Loss()
        self.loss = torch.nn.L1Loss()

    def forward(self, y_pred, y_true, x_input):
        temp_pred = self.loss_obj_input(y_pred, x_input)
        temp_true = self.loss_obj_input(y_true, x_input)
        return self.loss(temp_pred, temp_true)


class InputDistanceLossBiasfield(torch.nn.Module):
    # We are going to use this for bias fields where we predict both the homogeneous image as well as
    # the bias field
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, y_pred, y_true, x_input):
        recon_pred_input = (y_pred[:, 0] * y_pred[:, 1])[:, None]
        return self.loss(recon_pred_input, x_input)


class DiceLossSeb(torch.nn.Module):
    def forward(self, pred, target):
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.
    Return:
        the computed loss.
    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

    if not labels.dtype == torch.int64:
        raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
          used.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'mean', eps: Optional[float] = None) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: Optional[float] = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)



class DiceBCELoss(torch.nn.Module):
    def __init__(self, lambda_bce=1, lambda_dice=1):
        super().__init__()
        self.loss_obj = torch.nn.BCELoss()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice

    def forward(self, pred, target):
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        dice_loss = 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
        bce_loss = self.loss_obj(pred, target)
        #print('Output of dice loss / bce loss', dice_loss, bce_loss, self.lambda_bce)
        return self.lambda_dice * dice_loss + self.lambda_bce * bce_loss


class AbsL1Loss(torch.nn.Module):
    # Complex ABS
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        pred_abs = torch.sqrt(input[:, 0] ** 2 + input[:, 1] ** 2)
        tgt_abs = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2)
        return F.l1_loss(pred_abs, tgt_abs, reduction=self.reduction)


class AngleL1Loss(torch.nn.Module):
    # Complex angle
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        pred_angle = torch.atan2(input[:, 1], input[:, 0])
        tgt_angle = torch.atan2(target[:, 1], target[:, 0])

        return F.l1_loss(pred_angle, tgt_angle, reduction=self.reduction)


class FFTL1Loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.to_last = hlayer.SwapAxes2D(to_channel_last=True)
        self.abs_loss = AbsL1Loss()
        self.angle_loss = AngleL1Loss()

    def forward(self, input, target):
        pred_fft = torch.fft(self.to_last(input), signal_ndim=2)
        tgt_fft = torch.fft(self.to_last(target), signal_ndim=2)

        return (self.abs_loss(pred_fft, tgt_fft) + self.angle_loss(pred_fft, tgt_fft)) / \
               (self.abs_loss(pred_fft, tgt_fft) ** 2)


class WeightedCrossEntropyLoss(torch.nn.Module):
    """
    https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
    WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=0):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = torch.autograd.Variable(nominator / denominator, requires_grad=False)
        return class_weights


class PerceptualLoss(nn.Module):
    """
    Computes stuff
    """

    def __init__(self, vgg_model):
        super().__init__()
        self.vgg = vgg_model

    def forward(self, y_pred, y_true):
        # loss_obj_mse = torch.nn.MSELoss()
        loss_obj_l1 = torch.nn.L1Loss()

        loss = 0
        pred_model_split = torch.split(y_pred, 1, dim=1)
        true_model_split = torch.split(y_true, 1, dim=1)
        iter = 0

        for i_chan in range(len(pred_model_split)):
            # Added an extra no_grad clause
            with torch.no_grad():
                features_pred = self.vgg(torch.cat([pred_model_split[i_chan]], dim=1))
                features_target = self.vgg(torch.cat([true_model_split[i_chan]], dim=1))
            for x_ftr, y_ftr in zip(features_pred, features_target):
                iter += 1
                loss_value_ftr = loss_obj_l1(x_ftr, y_ftr)
                loss += loss_value_ftr
        loss /= iter
        return loss

    def _forward_debug(self, y_pred, y_true):
        loss_obj_mse = torch.nn.MSELoss()

        loss = 0
        pred_model_split = torch.split(y_pred, 1, dim=1)
        true_model_split = torch.split(y_true, 1, dim=1)
        iter = 0

        intermediate_features = []
        for i_chan in range(len(pred_model_split)):
            features_pred = self.vgg(torch.cat(3 * [pred_model_split[i_chan]], dim=1))
            features_target = self.vgg(torch.cat(3 * [true_model_split[i_chan]], dim=1))
            temp_feature = []
            for x_ftr, y_ftr in zip(features_pred, features_target):
                iter += 1
                temp_feature.append((x_ftr, y_ftr))
                loss_value_ftr = loss_obj_mse(x_ftr, y_ftr)
                loss += loss_value_ftr
            intermediate_features.append(temp_feature)

        loss /= iter
        return intermediate_features


class PerceptualLossStyleLoss(nn.Module):
    """
    Computes stuff
    """
    def __init__(self, vgg_model, alpha=0.5, **kwargs):
        super().__init__()
        self.vgg = vgg_model
        self.alpha = torch.nn.parameter.Parameter(torch.tensor(alpha), requires_grad=False)

    @staticmethod
    def gram_matrix(x_tens):
        """ Calculate the Gram Matrix of a given tensor
            Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        """

        # get the batch_size, depth, height, and width of the Tensor
        b, d, h, w = x_tens.size()
        # reshape so we're multiplying the features for each channel
        x_tens = x_tens.view(b, d, h * w)

        # calculate the gram matrix
        gram = torch.einsum("bcr, brd->bcd", x_tens, x_tens.transpose(-1, 1)) / (d * h * w)

        return gram

    def forward(self, y_pred, y_true):
        loss_obj_mse = torch.nn.MSELoss()
        frobenius_loss = FrobeniusLoss()

        loss = 0
        pred_model_split = torch.split(y_pred, 1, dim=1)
        true_model_split = torch.split(y_true, 1, dim=1)
        iter = 0

        for i_chan in range(len(pred_model_split)):
            features_pred = self.vgg(pred_model_split[i_chan])
            features_target = self.vgg(true_model_split[i_chan])
            for x_ftr, y_ftr in zip(features_pred, features_target):
                iter += 1
                loss_value_ftr = loss_obj_mse(x_ftr, y_ftr)
                gram_x_ftr = self.gram_matrix(x_ftr)
                gram_y_ftr = self.gram_matrix(y_ftr)
                style_loss = frobenius_loss(gram_x_ftr, gram_y_ftr)
                loss += self.alpha * loss_value_ftr + (1 - self.alpha) * style_loss
        loss /= iter
        return loss


class PerceptualLossStyleLossComplexLoss(nn.Module):
    """
    Computes stuff
    """
    def __init__(self, vgg_model, alpha=0.5, lambda_cos_angle=0.1, **kwargs):
        super().__init__()
        self.vgg = vgg_model
        self.mse_loss = nn.MSELoss()
        self.alpha = torch.nn.parameter.Parameter(torch.tensor(alpha), requires_grad=True)
        self.display_counter = 0
        self.max_display = 2
        self.lambda_cos_angle = lambda_cos_angle

    def _get_cos_angle(self, x):
        return torch.cos(torch.atan2(x[:, 1], x[:, 0]))

    @staticmethod
    def gram_matrix(x_tens):
        """ Calculate the Gram Matrix of a given tensor
            Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        """

        # get the batch_size, depth, height, and width of the Tensor
        b, d, h, w = x_tens.size()
        # reshape so we're multiplying the features for each channel
        x_tens = x_tens.view(b, d, h * w)

        # calculate the gram matrix
        gram = torch.einsum("bcr, brd->bcd", x_tens, x_tens.transpose(-1, 1)) / (d * h * w)

        return gram

    def forward(self, y_pred, y_true):
        loss_obj_mse = torch.nn.MSELoss()
        frobenius_loss = FrobeniusLoss()

        loss = 0
        pred_model_split = torch.split(y_pred, 2, dim=1)
        true_model_split = torch.split(y_true, 2, dim=1)
        iter = 0

        for i_chan in range(len(pred_model_split)):
            temp_pred = pred_model_split[i_chan]
            temp_target = true_model_split[i_chan]

            pred_angle = temp_pred[:, 0] ** 2 + temp_pred[:, 1] ** 2
            target_angle = temp_target[:, 0] ** 2 + temp_target[:, 1] ** 2

            features_pred = self.vgg(torch.stack(3 * [pred_angle], dim=1))
            features_target = self.vgg(torch.stack(3 * [target_angle], dim=1))

            for x_ftr, y_ftr in zip(features_pred, features_target):
                iter += 1
                loss_value_ftr = loss_obj_mse(x_ftr, y_ftr)
                gram_x_ftr = self.gram_matrix(x_ftr)
                gram_y_ftr = self.gram_matrix(y_ftr)
                style_loss = frobenius_loss(gram_x_ftr, gram_y_ftr)
                loss += self.alpha * loss_value_ftr + (1 - self.alpha) * style_loss

            if self.display_counter < self.max_display:
                print('Channel: ', i_chan)
                print(f'\t PerceptualLossStyleLossL1Loss: combined loss {loss}')

        loss /= iter
        if self.display_counter < self.max_display:
            print('Accumulated feature/style loss', loss)
        #
        # # This means that we can only use this one when we are dealing with complex numbers....
        # pred_model_split_chan = torch.split(y_pred, 2, dim=1)
        # true_model_split_chan = torch.split(y_true, 2, dim=1)
        #
        # temp_loss = 0
        # for i_chan in range(len(pred_model_split_chan)):
        #     temp_cos_angle_loss_pred = self._get_cos_angle(pred_model_split_chan[i_chan])
        #     temp_cos_angle_loss_target = self._get_cos_angle(true_model_split_chan[i_chan])
        #     temp_loss += self.mse_loss(temp_cos_angle_loss_pred, temp_cos_angle_loss_target)
        #     if self.display_counter < self.max_display:
        #         print('Channel: ', i_chan)
        #         print(f'\t PerceptualLossStyleLossL1Loss: cos angle loss {temp_loss} - ', self.lambda_cos_angle, self.alpha)
        #
        # temp_loss = temp_loss / len(pred_model_split_chan)
        # loss = (loss + self.lambda_cos_angle * temp_loss) / 2
        self.display_counter += 1
        return loss


class L1LossRelative(nn.Module):
    """
    Relative L1 loss with mask option
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, y_pred, y_true, mask=None):
        # The previous Relative loss.. was not so relative
        # A Relative loss should satisfy d(x, y) = d(lambda x, lambda y)
        if mask is None:
            return torch.mean((torch.abs(y_pred - y_true) / (torch.abs(y_true) + torch.abs(y_pred))))
        else:
            mask = mask.bool()[0]
            return torch.mean((torch.abs(y_pred[:, mask] - y_true[:, mask]) / (torch.abs(y_true[:, mask]) + torch.abs(y_pred[:, mask]))))


class L1LossSemiRelative(nn.Module):
    """
    The other loss exploded. Could be because of SUM instead of MEAN. But want to try this out as well and see the difference.
    """

    def forward(self, y_pred, y_true):
        # The previous Relative loss.. was not so relative
        # A Relative loss should satisfy d(x, y) = d(lambda x, lambda y)
        return torch.mean(torch.abs(y_pred - y_true) / (1 + torch.abs(y_true)))


class WeightedL1Loss(nn.Module):
    """
    The other loss exploded. Could be because of SUM instead of MEAN. But want to try this out as well and see the difference.
    """

    def forward(self, y_pred, y_true):
        # The previous Relative loss.. was not so relative
        # A Relative loss should satisfy d(x, y) = d(lambda x, lambda y)
        n_batch = y_true.shape[0]
        n_chan = y_true.shape[1]

        return torch.mean(torch.abs(y_pred - y_true))


class L1LossPerceptualLoss(nn.Module):
    """
    Combination of L1 loss and Perceptual loss.
    This is now (better) possible with the additional GAN loss
    """
    def __init__(self, vgg_model, alpha=1.0):
        super().__init__()
        self.vgg = vgg_model
        self.alpha = torch.nn.parameter.Parameter(torch.tensor(alpha), requires_grad=True)
        self.loss_obj_mse = torch.nn.MSELoss()
        self.loss_obj_l1 = torch.nn.L1Loss()

    def forward(self, y_pred, y_true):
        loss = 0
        pred_model_split = torch.split(y_pred, 1, dim=1)
        true_model_split = torch.split(y_pred, 1, dim=1)
        iter = 0

        for i_chan in range(len(pred_model_split)):
            features_pred = self.vgg(torch.cat(3 * [pred_model_split[i_chan]], dim=1))
            features_target = self.vgg(torch.cat(3 * [true_model_split[i_chan]], dim=1))
            for x_ftr, y_ftr in zip(features_pred, features_target):
                iter += 1
                loss_value_ftr = self.loss_obj_mse(x_ftr, y_ftr)
                loss += loss_value_ftr
        loss /= iter
        return loss + self.alpha * self.loss_obj_l1(y_pred, y_true)


class L1L2Loss(nn.Module):
    """
    Computes stuff
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.lambda_1 = torch.nn.parameter.Parameter(torch.tensor(alpha), requires_grad=True)
        self.lambda_2 = torch.nn.parameter.Parameter(torch.tensor(alpha), requires_grad=True)
        self.loss_obj_mse = torch.nn.MSELoss()
        self.loss_obj_l1 = torch.nn.L1Loss()

    def forward(self, y_pred, y_true):
        return self.lambda_1 * self.loss_obj_mse(y_pred, y_true) + self.lambda_2 * self.loss_obj_l1(y_pred, y_true)


class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        n_dmi = y_pred.ndim
        sel_axes = tuple(list(range(-(n_dmi-1), 0)))
        frob_norm = torch.sqrt((torch.abs(y_pred - y_true) ** 2).sum(dim=(sel_axes))).sum()
        return frob_norm


class ContextualLoss(nn.Module):
    """
    This computes CX(X, Y) where
    X and Y are sets of features.

    X = {x_1, ..., x_n} and Y = {y_1, ..., y_m},
    where x_i and y_j are spatial patches
    of features with shape [channels, size, size].

    It is assumed that Y is fixed and doesn't change!

    source: https://github.com/TropComplique/contextual-loss/blob/master/contextual_loss.py    """

    def __init__(self, y, size, stride, h, distance='cosine', ep=1e-10):
        """
        Arguments:
            y: a float tensor with shape [1, C, A, B].
            size, stride: integers, parameters of used patches.
            h: a float number.
            distance: a string, possible values are ['cosine', 'l2_squared'].
        """
        super().__init__()

        assert distance in ['cosine', 'l2_squared']
        normalize = distance == 'cosine'
        self.distance = distance

        y = y.squeeze(0)
        y_mu = torch.mean(y, dim=[1, 2], keepdim=True)  # shape [C, 1, 1]
        y = extract_patches(y - y_mu, size, stride, normalize)  # shape [M, C, size, size]

        self.y_mu = nn.Parameter(data=y_mu, requires_grad=False)
        self.y = nn.Parameter(data=y, requires_grad=False)

        self.stride = stride
        self.h = h

    def forward(self, x, epsilon=10e-10):
        """
        Arguments:
            x: a float tensor with shape [1, C, H, W].
        Returns:
            a float tensor with shape [].
        """
        x = x.squeeze(0)
        x = x - self.y_mu  # shape [C, H, W]

        if self.distance == 'cosine':
            similarity = cosine_similarity(x, self.y, self.stride)
            d = torch.clamp(1.0 - similarity, min=0.0, max=2.0)
        else:
            d = squared_l2_distance(x, self.y, self.stride)
            d = torch.clamp(d, min=0.0)

        # d has shape [N, M],
        # where N is the number of features on x
        # and M is the number of features on y

        d_min, _ = torch.min(d, dim=1, keepdim=True)
        # it has shape [N, 1]

        epsilon_from_the_paper = 1e-5
        d_tilde = d / (d_min + epsilon_from_the_paper)
        # it has shape [N, M]

        w = torch.exp(-d_tilde/self.h)  # shape [N, M]
        cx_ij = w / (torch.sum(w, dim=1, keepdim=True) + epsilon)
        # it has shape [N, M]

        max_i_cx_ij, _ = torch.max(cx_ij, dim=0)  # shape [M]
        cx = torch.mean(max_i_cx_ij, dim=0)  # shape []
        cx_loss = -torch.log(cx + epsilon)
        return cx_loss


class SmoothedComplex(nn.Module):
    def __init__(self, start_n=64):
        """
        Assuems input of the shape (batch, 2, ny, nx)
        Where the channel dimension contains the real/imag part resp.
        :param start_n:
        """
        super().__init__()
        self.n_mask = start_n
        self.loss_obj = nn.L1Loss()

    def smooth_input(self, x):
        kernel = np.ones((2, 1, self.n_mask, self.n_mask)) / self.n_mask ** 2
        kernel_tens = torch.from_numpy(kernel).float()
        device_tensor = x.device
        kernel_tens = kernel_tens.to(device_tensor)

        x_smoothed = F.conv2d(x, kernel_tens, groups=2)
        return x_smoothed

    def forward(self, y_pred, y_target):
        ypred_smoothed = self.smooth_input(y_pred)
        ytrgt_smoothed = self.smooth_input(y_target)

        # Could implement different complex variants here...
        ypred_angle = torch.atan2(ypred_smoothed[:, 1], ypred_smoothed[:, 0])
        ytrgt_angle = torch.atan2(ytrgt_smoothed[:, 1], ytrgt_smoothed[:, 0])
        return self.loss_obj(ypred_angle, ytrgt_angle)


class WeightedSmoothedComplex(nn.Module):
    def __init__(self, start_n=64):
        """
        Assuems input of the shape (batch, 2, ny, nx)
        Where the channel dimension contains the real/imag part resp.
        :param start_n:
        """
        super().__init__()
        self.n_mask = start_n
        self.loss_obj = nn.L1Loss()
        self.debug_display_counter = 0

    def smooth_input(self, x):
        kernel = np.ones((2, 1, self.n_mask, self.n_mask)) / self.n_mask ** 2
        kernel_tens = torch.from_numpy(kernel).float()
        device_tensor = x.device
        if self.debug_display_counter < 1:
            print('WeightedSmoothed Complex: putting kernel to device ', device_tensor)
            self.debug_display_counter += 1

        kernel_tens = kernel_tens.to(device_tensor)

        x_smoothed = F.conv2d(x, kernel_tens, groups=2)
        return x_smoothed

    def forward(self, y_pred, y_target):
        ypred_smoothed = self.smooth_input(y_pred)
        ytrgt_smoothed = self.smooth_input(y_target)

        # Could implement different complex variants here...
        ypred_abs = (ypred_smoothed[:, 1] ** 2 + ypred_smoothed[:, 0] ** 2)
        ypred_abs = ypred_abs / torch.max(ypred_abs)
        ypred_angle = torch.atan2(ypred_smoothed[:, 1], ypred_smoothed[:, 0]) * (ypred_abs ** 2)

        ytrgt_abs = (ytrgt_smoothed[:, 1] ** 2 + ytrgt_smoothed[:, 0] ** 2)
        ytrgt_abs = ytrgt_abs / torch.max(ytrgt_abs)
        ytrgt_angle = torch.atan2(ytrgt_smoothed[:, 1], ytrgt_smoothed[:, 0]) * (ytrgt_abs ** 2)

        return self.loss_obj(ypred_angle, ytrgt_angle)


class SmoothedComplexSplit(nn.Module):
    def __init__(self, start_n=64, n_split=8, **kwargs):
        """
        Assuems input of the shape (batch, 2, ny, nx)
        Where the channel dimension contains the real/imag part resp.
        :param start_n:
        """
        super().__init__()
        self.n_mask = start_n
        self.loss_obj = nn.L1Loss()
        self.smoothed_complex = SmoothedComplex(start_n)
        self.n_split = n_split

        self.internal_counter = 0
        self.interal_max = 20  # This is the amount of max iterations to be reached to continue to the next level
        self.epsilon = 1e-04  # Loss value that needs to be reached

    def forward(self, y_pred, y_target):
        # Create the smoothing mask.
        y_pred_split = torch.chunk(y_pred, self.n_split, dim=1)
        y_target_split = torch.chunk(y_target, self.n_split, dim=1)
        loss_list = [self.smoothed_complex(ypred, ytarget) for ypred, ytarget in zip(y_pred_split, y_target_split)]
        loss_tensor = torch.Tensor(loss_list)
        loss_value = torch.mean(loss_tensor)

        # 'Brain' or 'memory' of the function
        if loss_value.item() < self.epsilon:
            self.internal_counter += 1
        else:
            self.internal_counter -= 1

        if self.internal_counter >= self.interal_max:
            self.n_mask = self.n_mask // 2
            self.n_mask = max(self.n_mask, 1)
            self.internal_counter = 0

        if self.internal_counter < -self.interal_max:
            self.n_mask = int(self.n_mask * 2)
            self.n_mask = min(self.n_mask, 128)
            self.internal_counter = 0

        return loss_value


class WeightedSmoothedComplexSplit(nn.Module):
    def __init__(self, start_n=64, n_split=8, **kwargs):
        """
        Assuems input of the shape (batch, 2, ny, nx)
        Where the channel dimension contains the real/imag part resp.
        :param start_n:
        """
        super().__init__()
        self.n_mask = start_n
        self.loss_obj = nn.L1Loss()
        self.smoothed_complex = WeightedSmoothedComplex(start_n)
        self.n_split = n_split

        self.internal_counter = 0
        self.interal_max = 20  # This is the amount of max iterations to be reached to continue to the next level
        self.epsilon = 1e-04  # Loss value that needs to be reached

    def forward(self, y_pred, y_target):
        # Create the smoothing mask.
        y_pred_split = torch.chunk(y_pred, self.n_split, dim=1)
        y_target_split = torch.chunk(y_target, self.n_split, dim=1)
        loss_list = [self.smoothed_complex(ypred, ytarget) for ypred, ytarget in zip(y_pred_split, y_target_split)]
        loss_tensor = torch.Tensor(loss_list)
        loss_value = torch.mean(loss_tensor)

        # 'Brain' or 'memory' of the function
        if loss_value.item() < self.epsilon:
            self.internal_counter += 1
        else:
            self.internal_counter -= 1

        if self.internal_counter >= self.interal_max:
            self.n_mask = self.n_mask // 2
            self.n_mask = max(self.n_mask, 1)
            self.internal_counter = 0

        if self.internal_counter < -self.interal_max:
            self.n_mask = int(self.n_mask * 2)
            self.n_mask = min(self.n_mask, 128)
            self.internal_counter = 0

        return loss_value


class ComplexPerceptualStyleSmoothedLnLoss(nn.Module):
    def __init__(self, vgg_model, complex_type, n_split, epsilon=1e-4, **kwargs):
        super().__init__()
        self.vgg = vgg_model
        self.ln_loss = LNloss()
        self.n_split = n_split
        self.epsilon = epsilon

        self.complex_type = complex_type

        self.interal_max = 20
        self.n_mask = 64
        self.internal_counter = 0

    def get_angle(self, x):
        return torch.atan2(x[:, 1], x[:, 0])

    def get_abs(self, x):
        return torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)

    def get_cos_angle(self, x):
        return torch.cos(torch.atan2(x[:, 1], x[:, 0]))

    @staticmethod
    def gram_matrix(x_tens):
        """ Calculate the Gram Matrix of a given tensor
            Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        """

        # get the batch_size, depth, height, and width of the Tensor
        b, d, h, w = x_tens.size()
        # reshape so we're multiplying the features for each channel
        x_tens = x_tens.view(b, d, h * w)

        # calculate the gram matrix
        gram = torch.einsum("bcr, brd->bcd", x_tens, x_tens.transpose(-1, 1)) / (d * h * w)

        return gram

    def forward(self, y_pred, y_target):

        # Smooth the prediction
        kernel = np.ones((2, 1, self.n_mask, self.n_mask)) / self.n_mask ** 2
        kernel_tens = torch.from_numpy(kernel).float()
        device_tensor = y_pred.device
        kernel_tens = kernel_tens.to(device_tensor)

        y_pred_split = torch.chunk(y_pred, self.n_split, dim=1)
        y_pred_smoothed = [F.conv2d(x, kernel_tens, groups=2) for x in y_pred_split]

        # Smooth the target
        y_target_split = torch.chunk(y_target, self.n_split, dim=1)
        y_target_smoothed = [F.conv2d(x, kernel_tens, groups=2) for x in y_target_split]

        # Transform to complex type -- We assume that we have real/imag in 0/1 channel per item
        if self.complex_type == 'cos_angle':
            pred_model_split = [self.get_cos_angle(x) for x in y_pred_smoothed]
            true_model_split = [self.get_cos_angle(x) for x in y_target_smoothed]
        elif self.complex_type == 'angle':
            pred_model_split = [self.get_angle(x) for x in y_pred_smoothed]
            true_model_split = [self.get_angle(x) for x in y_target_smoothed]
        elif self.complex_type == 'abs':
            pred_model_split = [self.get_abs(x) for x in y_pred_smoothed]
            true_model_split = [self.get_abs(x) for x in y_target_smoothed]
        elif self.complex_type == 'real':
            pred_model_split = [x[:, 0] for x in y_pred_smoothed]
            true_model_split = [x[:, 0] for x in y_target_smoothed]
        elif self.complex_type == 'imag':
            pred_model_split = [x[:, 1] for x in y_pred_smoothed]
            true_model_split = [x[:, 1] for x in y_target_smoothed]
        else:
            pred_model_split = []
            true_model_split = []

        # Calculate Perc/Style loss
        frobenius_loss = FrobeniusLoss()

        loss_value = 0
        iter = 0

        for i_chan in range(self.n_split):
            # Now convert to complex type?
            features_pred = self.vgg(torch.stack(3 * [pred_model_split[i_chan]], dim=1))
            features_target = self.vgg(torch.stack(3 * [true_model_split[i_chan]], dim=1))
            for x_ftr, y_ftr in zip(features_pred, features_target):
                iter += 1
                loss_value_ftr = self.ln_loss(x_ftr, y_ftr)
                gram_x_ftr = self.gram_matrix(x_ftr)
                gram_y_ftr = self.gram_matrix(y_ftr)
                style_loss = frobenius_loss(gram_x_ftr, gram_y_ftr)
                # REmoved the alpha... just using them both equally
                loss_value += 0.5 * loss_value_ftr + 0.5 * style_loss
        loss_value /= iter

        # 'Brain' or 'memory' of the function
        if loss_value.item() < self.epsilon:
            self.internal_counter += 1
        else:
            self.internal_counter -= 1

        if self.internal_counter >= self.interal_max:
            self.n_mask = self.n_mask // 2
            self.n_mask = max(self.n_mask, 1)
            self.internal_counter = 0

        if self.internal_counter < -self.interal_max:
            self.n_mask = int(self.n_mask * 2)
            self.n_mask = min(self.n_mask, 128)
            self.internal_counter = 0

        return loss_value


class LNloss(torch.nn.Module):
    # Trying out some fancy losses... hopefully stable enough
    # Calcualtes the L_n loss, and uses a different n each time.

    def __init__(self, start_n=16, epsilon=1e-04, interal_max=20, **kwargs):
        super().__init__()
        self.n = start_n  # (2 * start_n)
        self.start_n = start_n  # (2 * start_n)
        self.internal_counter = 0
        self.interal_max = interal_max  # This is the amount of max iterations to be reached to continue to the next level
        self.epsilon = epsilon  # Loss value that needs to be reached

    def forward(self, ypred, ytarget):
        n_power = self.n
        batch_dim = ypred.shape[0]
        n_pixels = np.prod(ypred.shape[-2:])
        temp_diff = (ypred - ytarget)
        scale_max = torch.abs(temp_diff).max()
        # print('Debug LNLoss scale max', scale_max)
        temp_diff = temp_diff / scale_max
        loss_value = (((torch.abs(temp_diff) ** n_power).T.sum_to_size(batch_dim)) ** (1/n_power)) / n_pixels
        # print('Debug LNloss loss value', loss_value)

        # 'Brain' or 'memory' of the function
        if torch.mean(loss_value).item() < self.epsilon:
            # print('mean loss value and epsilon', torch.mean(loss_value).item(), self.epsilon)
            self.internal_counter += 1
        else:
            self.internal_counter -= 1

        if self.internal_counter >= self.interal_max:
            self.n = self.n - 1
            self.internal_counter = 0

        if self.internal_counter < -self.interal_max:
            self.n = self.n + 1
            self.internal_counter = 0

        if self.n <= 0:
            self.n = self.start_n

        return torch.mean(loss_value)


class SpacyLoss(torch.nn.Module):
    """
    Calculates a potential SPACY decomposition.. and checks that to the closests "real" spacy prediction
    """
    def __init__(self, n_size, n_degree, transform_type, x_range=(-1, 1), y_range=(-1, 1),
                 epsilon_r=50, sigma=0.6, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)
        self.coil_position = kwargs.get('coil_position', False)

        # Physical parameters
        epsilon_0 = 8.8 * 1e-12
        mu_0 = 4 * np.pi * 1e-7
        omega = 42.58 * 7
        self.complex_wave_number = np.sqrt(mu_0 * epsilon_0 * epsilon_r * omega ** 2 + 1j * omega * sigma)

        self.n_size = n_size
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range

        self.loss = torch.nn.MSELoss()
        if self.coil_position:
            self.n_size = 2 * self.n_size

        temp_spacy = self.get_spacy_matrix(n_degree)

        if self.coil_position:
            coil_position = self.get_coil_position()
            n_equations = temp_spacy.shape[-1]
            temp_spacy_reshp = temp_spacy.reshape((2 * n_size, 2 * n_size, n_equations))

            temp_spacy_list = []
            for sel_x, sel_y in coil_position:
                # print(sel_x, sel_y)
                # print(n_size - sel_x, 2 * n_size - sel_x, n_size - sel_y, 2 * n_size - sel_y)
                temp_subset = temp_spacy_reshp[n_size - sel_x: 2 * n_size - sel_x, n_size - sel_y: 2 * n_size - sel_y]
                temp_spacy_list.append(temp_subset.reshape((-1, n_equations)))

            temp_spacy = np.concatenate(temp_spacy_list, axis=1)

        if transform_type == 'real':
            spacy_matrix = torch.as_tensor(temp_spacy.real).float()
        elif transform_type == 'imag':
            spacy_matrix = torch.as_tensor(temp_spacy.imag).float()
        else:
            spacy_matrix = None
            print('Unkown transform type in spacy loss', transform_type)

        self.spacy_matrix = torch.nn.Parameter(spacy_matrix, requires_grad=False)
        self.spacy_matrix_pinv = torch.nn.Parameter(torch.pinverse(self.spacy_matrix), requires_grad=False)

    @staticmethod
    def get_coil_position():
        # Average coil positoin calculated over many coils..
        # Obtained via `get_average_coil_position.py` script
        average_coil_pos = np.array(
            [[82.11470037, 239.8338015],
             [48.88810861, 169.99656679],
             [47.68789014, 93.52621723],
             [71.42993134, 20.5744382],
             [207.14325843, 16.53792135],
             [225.85549313, 86.63623596],
             [226.76186017, 161.39138577],
             [216.22487516, 235.51513733]]
        ).astype(int)
        return average_coil_pos

    def get_spacy_matrix(self, n_degree):
        print('\n\nCreating SPACY matrix. This might take some minutes...')

        x_range = np.linspace(self.x0, self.x1, self.n_size)
        y_range = np.linspace(self.y0, self.y1, self.n_size)
        X, Y = np.meshgrid(x_range, y_range)
        r = np.sqrt(X ** 2 + Y ** 2)
        # Theta: Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.
        theta = np.zeros(r.shape)
        # Phi : Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
        phi = np.arctan2(Y, X)

        theta[phi < 0] = np.pi
        phi = np.abs(phi)

        res = []
        # Shherical part...
        for i_degree in range(n_degree):
            jn = scipy.special.spherical_jn(n=i_degree, z=self.complex_wave_number * r)
            for m_order in range(-i_degree, i_degree):
                # Spherical part...
                temp_sph = jn * orthopoly.spherical_harmonic.sph_har(t=phi, p=theta, n=i_degree, m=m_order)
                res.append(temp_sph)

        for i_degree in range(-n_degree, n_degree):
            # Cylindrical part - based on Mikes expansion
            temp_cyl = scipy.special.jn(i_degree, self.complex_wave_number * r) * np.exp(1j * i_degree * phi)
            res.append(temp_cyl)

        n_equations = len(res)
        temp_A = np.array(res).reshape(n_equations, -1).T

        return temp_A

    def __call__(self, y_pred, y, mask=None):
        # y_pred is the... prediction
        if y_pred.ndim == 3:
            n_batch = y_pred.shape[0]
            n_chan = 1
        elif y_pred.ndim == 4:
            n_batch = y_pred.shape[0]
            n_chan = y_pred.shape[1]
        else:
            print(f'Unknown dimension from y_pred {y_pred.ndim}')
            n_batch = n_chan = None

        if self.debug:
            print(f'Using nchan {n_chan}')
            print(f'Using nbatch {n_batch}')

        y_pred_ravel = y_pred.reshape((n_batch, n_chan, -1)).T
        y_ravel = y.reshape((n_batch, n_chan, -1)).T
        if mask is not None:
            mask_ravel = mask.reshape((n_batch, n_chan, -1)).T

        if self.debug:
            print('y pred ravel', y_pred_ravel.shape)
            print('y ravel', y_ravel.shape)
            if mask is not None:
                print('mask ravel', mask_ravel.shape)
            print(' Shape matrix pinv ', self.spacy_matrix_pinv.shape)
            print(' Shape matrix ', self.spacy_matrix.shape)

        cum_loss = 0
        counter = 0
        # It is too annoying to cahgne verything to broadcast proeprylll
        # So we are just oging to loop over the batches as well
        for i_batch in range(n_batch):
            for i_chan in range(n_chan):
                y_pred_temp = y_pred_ravel[:, i_chan, i_batch]
                y_temp = y_ravel[:, i_chan, i_batch]
                if mask is not None:
                    spacy_mask_inv = self.spacy_matrix_pinv[:, mask_ravel[:, i_chan, i_batch] == 1]
                    spacy_mask = self.spacy_matrix[mask_ravel[:, i_chan, i_batch] == 1]
                    y_pred_ravel_mask = y_pred_temp[mask_ravel[:, i_chan, i_batch] == 1]
                    y_ravel_mask = y_temp[mask_ravel[:, i_chan, i_batch] == 1]

                    x_pred_tensor = torch.matmul(spacy_mask_inv, y_pred_ravel_mask)
                    x_tensor = torch.matmul(spacy_mask_inv, y_ravel_mask)

                    b_pred_masked = torch.matmul(spacy_mask, x_pred_tensor)
                    b_masked = torch.matmul(spacy_mask, x_tensor)

                    temp_device = b_pred_masked.get_device()
                    b_pred = torch.zeros(y_pred_temp.shape)
                    b = torch.zeros(y_pred_temp.shape)

                    b_pred = b_pred.to(temp_device)
                    b = b.to(temp_device)
                    # print('Here we run into a problem..')
                    # print('bpred device ', b_pred.get_device())
                    # print('b device ', b.get_device())
                    # print('bpred mask device ', b_pred_masked.get_device())
                    # print('\t  spacy mask ', spacy_mask.get_device())
                    # print('\t  xpredtensor mask ', x_pred_tensor.get_device())
                    # print('b mask device ', b_masked.get_device())
                    # print('\t  x tensor mask ', x_tensor.get_device())
                    # print('Mask ravel device', mask_ravel.get_device())

                    b_pred[mask_ravel[:, i_chan, i_batch] == 1] = b_pred_masked
                    b[mask_ravel[:, i_chan, i_batch] == 1] = b_masked
                else:

                    x_pred_tensor = torch.matmul(self.spacy_matrix_pinv, y_pred_temp)
                    x_tensor = torch.matmul(self.spacy_matrix_pinv, y_temp)
                    b_pred = torch.matmul(self.spacy_matrix, x_pred_tensor)
                    b = torch.matmul(self.spacy_matrix, x_tensor)

                    # print('Here we dont run into a problem')
                    # print('bpred device ', b_pred.get_device())
                    # print('b device ', b.get_device())
                    # print('\t  xpredtensor mask ', x_pred_tensor.get_device())
                    # print('\t  x tensor mask ', x_tensor.get_device())
                    # print('spacy matrix ', self.spacy_matrix.get_device())
                    # print('spacy inv matrix ', self.spacy_matrix_pinv.get_device())

                loss_item = self.loss(b_pred, b)

                cum_loss += loss_item
                counter += 1

        return cum_loss / counter

    def test_call(self, y_pred, y, mask=None):
        # y_pred is the... prediction
        if y_pred.ndim == 3:
            n_batch = y_pred.shape[0]
            n_chan = 1
        elif y_pred.ndim == 4:
            n_batch = y_pred.shape[0]
            n_chan = y_pred.shape[1]
        else:
            print(f'Unknown dimension from y_pred {y_pred.ndim}')
            n_batch = n_chan = None

        y_pred_ravel = y_pred.reshape((n_batch, n_chan, -1)).T
        y_ravel = y.reshape((n_batch, n_chan, -1)).T
        if mask is not None:
            mask_ravel = mask.reshape((n_batch, n_chan, -1)).T

        channel_list = []
        cum_loss = 0
        for i_batch in range(n_batch):
            for i_chan in range(n_chan):
                y_pred_temp = y_pred_ravel[:, i_chan, i_batch]
                y_temp = y_ravel[:, i_chan, i_batch]
                if mask is not None:
                    spacy_mask_inv = self.spacy_matrix_pinv[:, mask_ravel[:, i_chan, i_batch] == 1]
                    spacy_mask = self.spacy_matrix[mask_ravel[:, i_chan, i_batch] == 1]
                    y_pred_ravel_mask = y_pred_temp[mask_ravel[:, i_chan, i_batch] == 1]
                    y_ravel_mask = y_temp[mask_ravel[:, i_chan, i_batch] == 1]

                    x_pred_tensor = torch.matmul(spacy_mask_inv, y_pred_ravel_mask)
                    x_tensor = torch.matmul(spacy_mask_inv, y_ravel_mask)

                    b_pred_masked = torch.matmul(spacy_mask, x_pred_tensor)
                    b_masked = torch.matmul(spacy_mask, x_tensor)

                    b_pred = torch.zeros(y_pred_temp.shape)
                    b = torch.zeros(y_pred_temp.shape)
                    b_pred[mask_ravel[:, i_chan, i_batch] == 1] = b_pred_masked
                    b[mask_ravel[:, i_chan, i_batch] == 1] = b_masked
                else:
                    x_pred_tensor = torch.matmul(self.spacy_matrix_pinv, y_pred_temp)
                    x_tensor = torch.matmul(self.spacy_matrix_pinv, y_temp)
                    b_pred = torch.matmul(self.spacy_matrix, x_pred_tensor)
                    b = torch.matmul(self.spacy_matrix, x_tensor)

                channel_list.append((b_pred, b))

        return cum_loss, channel_list


class ShimLoss(torch.nn.Module):
    def __init__(self, return_type='mse'):
        super().__init__()
        return_type = return_type
        return_dict = {'mse': self.mse_loss,
                       'max_mse': self.max_mse,
                       'max_ge': self.max_ge,
                       'max_hom': self.max_hom,
                       'max_fa': self.max_fa}
        self.return_fun = return_dict[return_type]

    @staticmethod
    def mse_loss(target, result, mask=None):
        loss_mse = nn.MSELoss()
        if mask is None:
            loss_value = loss_mse(target, result)
        else:
            loss_value = loss_mse(target * mask, result * mask)
        return loss_value

    @staticmethod
    def max_mse(target, result, mask=None):
        if mask is None:
            loss_value = -torch.mean((result) ** 2)
        else:
            loss_value = -torch.mean((result * mask) ** 2)
        return loss_value

    @staticmethod
    def max_hom(target ,result ,mask=None):
        if mask is None:
            loss_value = - torch.std(result) / torch.mean(result)
        else:
            loss_value = - torch.std(result * mask) / torch.mean(result * mask)
        return loss_value

    @staticmethod
    def max_fa(target ,result ,mask=None):
        if mask is None:
            loss_value = -torch.mean(result) / - torch.std(result)
        else:
            loss_value = -torch.mean(result * mask) / - torch.std(result * mask)
        return loss_value

    @staticmethod
    def max_ge(target, result, mask=None):
        if mask is None:
            signal = torch.sin(torch.abs(result))
            loss_value = -torch.mean(signal) / torch.std(signal)
        else:
            signal = torch.sin(torch.abs(result * mask))
            loss_value = -torch.mean(signal) / torch.std(signal)
        return loss_value

    @staticmethod
    def apply_transmit_phase(input_tensor_real, input_tensor_imag, phase_prediction):
        phase_sin = torch.diag_embed(torch.sin(phase_prediction))
        phase_cos = torch.diag_embed(torch.cos(phase_prediction))

        # temp_input_cos = torch.einsum("bxytr, bts -> bxysr", input_tensor_real, phase_cos)
        # temp_input_sin = torch.einsum("bxytr, bts -> bxysr", input_tensor_imag, phase_sin)
        temp_input_cos = torch.einsum("btrxy, bts -> bsrxy", input_tensor_real, phase_cos)
        temp_input_sin = torch.einsum("btrxy, bts -> bsrxy", input_tensor_imag, phase_sin)
        input_real = temp_input_cos - temp_input_sin

        # temp_input_cos = torch.einsum("bxytr, bts -> bxysr", input_tensor_imag, phase_cos)
        # temp_input_sin = torch.einsum("bxytr, bts -> bxysr", input_tensor_real, phase_sin)
        temp_input_cos = torch.einsum("btrxy, bts -> bsrxy", input_tensor_imag, phase_cos)
        temp_input_sin = torch.einsum("btrxy, bts -> bsrxy", input_tensor_real, phase_sin)
        input_imag = temp_input_cos + temp_input_sin

        return input_real, input_imag

    @staticmethod
    def apply_receive_phase(input_tensor_real, input_tensor_imag, phase_prediction):
        phase_sin = torch.diag_embed(torch.sin(phase_prediction))
        phase_cos = torch.diag_embed(torch.cos(phase_prediction))

        # Calculate the real components with the obtained prediction results...
        # temp_input_cos = torch.einsum("bxytr, brs -> bxyts", input_tensor_real, phase_cos)
        # temp_input_sin = torch.einsum("bxytr, brs -> bxyts", input_tensor_imag, phase_sin)
        temp_input_cos = torch.einsum("btrxy, brs -> btsxy", input_tensor_real, phase_cos)
        temp_input_sin = torch.einsum("btrxy, brs -> btsxy", input_tensor_imag, phase_sin)
        input_real = temp_input_cos - temp_input_sin

        # temp_input_cos = torch.einsum("bxytr, brs -> bxyts", input_tensor_imag, phase_cos)
        # temp_input_sin = torch.einsum("bxytr, brs -> bxyts", input_tensor_real, phase_sin)
        temp_input_cos = torch.einsum("btrxy, brs -> btsxy", input_tensor_imag, phase_cos)
        temp_input_sin = torch.einsum("btrxy, brs -> btsxy", input_tensor_real, phase_sin)
        input_imag = temp_input_cos + temp_input_sin

        return input_real, input_imag

    @staticmethod
    def process_input(input):
        input_real, input_imag = torch.split(input, 64, dim=1)
        # print(input_real.shape, input_imag.shape) --> (None, 50, 50, 64), (None, 50, 50, 64)
        input_shape = input_real.shape
        image_size = input_shape[-2:]  # --> (50, 50)
        # image_size = input_shape[:2]
        new_shape = (-1, 8, 8, *image_size)

        real_input_tensor = torch.reshape(input_real, new_shape)
        imag_input_tensor = torch.reshape(input_imag, new_shape)

        return real_input_tensor, imag_input_tensor

    def __call__(self, prediction, target, input, mask=None):
        real_input_tensor, imag_input_tensor = self.process_input(input)

        abs_input = torch.sqrt(real_input_tensor ** 2 + imag_input_tensor ** 2)
        # (x, y, t_c, r_c)
        # target_tensor = torch.sum(torch.sum(abs_input, dim=1), dim=1)
        # (x, y) -> absolute waarde is ||z||

        # Voorspellingen gebruiken op de input
        # Prediction is van de formaat (batch_size, 16)
        prediction_transmit = prediction[:, :8]
        prediction_receive = prediction[:, 8:]

        # Calculate the effect of the transmit phase setting on the input
        tx_input_real, tx_input_imag = self.apply_transmit_phase(real_input_tensor, imag_input_tensor, prediction_transmit)

        # On top of that, apply the receive phase setting
        rx_tx_input_real, rx_tx_input_imag = self.apply_receive_phase(tx_input_real, tx_input_imag, prediction_receive)

        # Now calculate the magnitude. We have succesfully avoided any complex arguments
        real_sum = torch.sum(torch.sum(rx_tx_input_real, dim=1), dim=1)
        imag_sum = torch.sum(torch.sum(rx_tx_input_imag, dim=1), dim=1)
        result = torch.sqrt(real_sum ** 2 + imag_sum ** 2)

        # result = tf.sqrt(rx_tx_input_real ** 2 + rx_tx_input_imag ** 2)
        # result_summed = torch.sum(torch.sum(result, dim=-1), dim=-1)

        # Here we calculate the loss between the 'shimmed' image and the absolute target

        if mask is None:
            return self.return_fun(target, result)
        else:
            return self.return_fun(target, result, mask)

    def debug_call(self, prediction, target, input):
        real_input_tensor, imag_input_tensor = self.process_input(input)

        real_sum_interf = torch.mean(torch.mean(real_input_tensor, dim=1), dim=1)
        imag_sum_interf = torch.mean(torch.mean(imag_input_tensor, dim=1), dim=1)
        result_interf = torch.sqrt(real_sum_interf ** 2 + imag_sum_interf ** 2)

        abs_input = torch.sqrt(real_input_tensor ** 2 + imag_input_tensor ** 2)
        # (x, y, t_c, r_c)
        target_tensor = torch.mean(torch.mean(abs_input, dim=1), dim=1)
        # (x, y) -> absolute waarde is ||z||

        # Voorspellingen gebruiken op de input
        # Prediction is van de formaat (batch_size, 16)
        prediction_transmit = prediction[:, :8]
        prediction_receive = prediction[:, 8:]

        # Calculate the effect of the transmit phase setting on the input
        tx_input_real, tx_input_imag = self.apply_transmit_phase(real_input_tensor, imag_input_tensor,
                                                                 prediction_transmit)

        # On top of that, apply the receive phase setting
        rx_tx_input_real, rx_tx_input_imag = self.apply_receive_phase(tx_input_real, tx_input_imag, prediction_receive)

        # Now calculate the magnitude. We have succesfully avoided any complex arguments
        real_sum = torch.mean(torch.mean(rx_tx_input_real, dim=1), dim=1)
        imag_sum = torch.mean(torch.mean(rx_tx_input_imag, dim=1), dim=1)
        result = torch.sqrt(real_sum ** 2 + imag_sum ** 2)

        return {'target': target_tensor, 'result': result, 'result_interf': result_interf}


class AverageZero(torch.nn.Module):
    def __init__(self, max_value=0.0):
        super().__init__()
        self.max_value = torch.nn.Parameter(torch.Tensor([max_value]), requires_grad=False)
        self.loss = torch.nn.L1Loss()

    def __call__(self, prediction, target):
        n_dim = prediction.ndim
        mean_pred = torch.mean(prediction, dim=tuple(range(-n_dim+1, 0, 1)))
        return self.loss(mean_pred, self.max_value)


class BalancedAveragedHausdorffLoss(nn.Module):
    # This should in theory be able to handle batch/channel predictions.
    def __init__(self, atol=0.3, n_norm=2, **kwargs):
        super().__init__()
        self.n_norm = n_norm
        self.atol = atol

    def forward(self, pred, target):
        """
        Compute the Averaged Hausdorff Distance function
        between two unordered sets of points (the function is symmetric).

        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """

        # Make sure that we are dealing with a binary image.
        pred = torch.isclose(pred, torch.tensor(1.), atol=self.atol).float()
        pred_shape = pred.shape
        xy_shape = pred_shape[-2:]
        n_batch_chan = torch.prod(torch.tensor(pred_shape[:2]))

        # Reshape pred and target into (-1, x, y)
        pred = pred.reshape((-1, *xy_shape))
        target = target.reshape((-1, *xy_shape))
        batch_pred_loc, x_pred_loc, y_pred_loc = torch.where(pred)
        pred_loc = torch.stack([x_pred_loc, y_pred_loc], dim=-1)
        batch_tgt_loc, x_tgt_loc, y_tgt_loc = torch.where(target)
        tgt_loc = torch.stack([x_tgt_loc, y_tgt_loc], dim=-1)

        loss_value = 0
        for i in range(0, n_batch_chan):
            n_points_ground_truth = torch.sum(batch_tgt_loc == i)
            differences = pred_loc[batch_pred_loc == i].unsqueeze(1) - tgt_loc[batch_tgt_loc == i].unsqueeze(0)
            distances = torch.sum(torch.abs(differences.float()) ** 2, -1) ** (1/2)

            #if len(distances):
            if (distances.shape[0] == 0) or (distances.shape[1] == 0):
                # Modified Chamfer Loss
                term_1 = 0
                term_2 = 0
            else:
                term_1 = torch.sum(torch.min(distances, 1)[0]) / n_points_ground_truth
                term_2 = torch.sum(torch.min(distances, 0)[0]) / n_points_ground_truth

            loss_value += (term_1 + term_2) / 2

        loss_value = loss_value / n_batch_chan

        return loss_value


class EfficientBalancedAveragedHausdorffLoss(nn.Module):
    # This should in theory be able to handle batch/channel predictions.
    def __init__(self, atol=0.3, n_norm=2, **kwargs):
        super().__init__()
        self.n_norm = n_norm
        self.atol = atol
        self.counter = 0
        self.debug = False

    @staticmethod
    def calculate(x, y):
        x_shape = x.shape
        xy_shape = x_shape[-2:]
        n_batch_chan = torch.prod(torch.tensor(x_shape[:2]))

        # Reshape pred and target into (-1, x, y)
        pred = x.reshape((-1, *xy_shape))
        target = y.reshape((-1, *xy_shape))
        batch_pred_loc, x_pred_loc, y_pred_loc = torch.where(pred)
        pred_loc = torch.stack([x_pred_loc, y_pred_loc], dim=-1)
        batch_tgt_loc, x_tgt_loc, y_tgt_loc = torch.where(target)
        tgt_loc = torch.stack([x_tgt_loc, y_tgt_loc], dim=-1)

        loss_value = 0
        for i in range(0, n_batch_chan):
            n_points_ground_truth = torch.sum(batch_tgt_loc == i)
            differences = pred_loc[batch_pred_loc == i][:1000].unsqueeze(1) - tgt_loc[batch_tgt_loc == i][:1000].unsqueeze(0)
            distances = torch.sum(torch.abs(differences.float()) ** 2, -1) ** (1 / 2)
#            print('Points', n_points_ground_truth)
#            print('Differences ', differences.shape)
#            print('Distances ', distances.shape)

            # if len(distances):
            if (distances.shape[0] == 0) or (distances.shape[1] == 0):
                # Modified Chamfer Loss
                term_1 = 0
                term_2 = 0
            else:
                min_distances, min_indices = torch.min(distances, 1)
         #       print('n_points_ground_truth', n_points_ground_truth)
          #      print('Min distances ', min_distances.shape, min_distances)
                term_1 = torch.sum(min_distances) / n_points_ground_truth
                # term_2 = torch.sum(torch.min(distances, 0)[0]) / n_points_ground_truth

            loss_value += term_1
        loss_value = loss_value / n_batch_chan

        return loss_value

    def forward(self, pred, target):
        """
        Compute the Averaged Hausdorff Distance function
        between two unordered sets of points (the function is symmetric).

        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """
        # Make sure that we are dealing with a binary image.
        pred = torch.isclose(pred, torch.tensor(1.), atol=self.atol).float()
        target = torch.isclose(target, torch.tensor(1.), atol=self.atol).float()

        #print('Debugging this loss')
        pred_target_diff_mask = (pred - target) == 1
        if self.debug:
            print('Equal to 1, target diff', (pred_target_diff_mask == 1).sum())
            print('Equal to 0, target diff', (pred_target_diff_mask == 0).sum())
            print('Equal to 1, target', (target == 1).sum())
            print('Equal to 0, target', (target == 0).sum())

        loss_value_1 = self.calculate(pred_target_diff_mask, target)
        #import helper.plot_class as hplotc
        #plot_obj = hplotc.ListPlot([pred_target_diff_mask.detach().cpu().numpy(), target.detach().cpu().numpy()])
        #self.counter += 1
        #file_name = f'/data/seb/pred_target_diff_target_{self.counter}.png'
        #plot_obj.figure.savefig(file_name)

        target_pred_diff_mask = (target - pred) == 1
        if self.debug:
            print('Equal to 1, pred diff', (target_pred_diff_mask == 1).sum())
            print('Equal to 0, pred diff', (target_pred_diff_mask == 0).sum())
            print('Equal to 1, pred', (pred == 1).sum())
            print('Equal to 0, pred', (pred == 0).sum())
        loss_value_2 = self.calculate(target_pred_diff_mask, pred)
        return (loss_value_1 + loss_value_2) / 2


def simplex(t: Tensor, axis=1) -> bool:
        _sum = cast(Tensor, t.sum(axis).type(torch.float32))
        _ones = torch.ones_like(_sum, dtype=torch.float32)
        return torch.allclose(_sum, _ones)


class CrossEntropy:
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.nd: str = kwargs["nd"]
        print(f"> Initialized {self.__class__.__name__} with kwargs:")
        hmisc.print_dict(kwargs)

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor, ___) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        loss = - einsum(f"bk{self.nd},bk{self.nd}->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class AbstractConstraints:
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.nd: str = kwargs["nd"]
        self.C = len(self.idc)
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"> Initialized {self.__class__.__name__} with kwargs:")
        hmisc.print_dict(kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        """
        id: int - Is used to tell if is it the upper or the lower bound
                  0 for lower, 1 for upper
        """
        raise NotImplementedError

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor, filenames: List[str]) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        assert probs.shape == target.shape

        # b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        b: int
        b, _, *im_shape = probs.shape
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2

        value: Tensor = cast(Tensor, self.__fn__(probs[:, self.idc, ...]))
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        upper_z: Tensor = cast(Tensor, (value - upper_b).type(torch.float32)).reshape(b, self.C * k)
        lower_z: Tensor = cast(Tensor, (lower_b - value).type(torch.float32)).reshape(b, self.C * k)
        assert len(upper_z) == len(lower_b) == len(filenames)

        upper_penalty: Tensor = self.penalty(upper_z)
        lower_penalty: Tensor = self.penalty(lower_z)
        assert upper_penalty.numel() == lower_penalty.numel() == upper_z.numel() == lower_z.numel()

        # f for flattened axis
        res: Tensor = einsum("f->", upper_penalty) + einsum("f->", lower_penalty)

        loss: Tensor = res.sum() / reduce(mul, im_shape)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss


class NaivePenalty(AbstractConstraints):
    def penalty(self, z: Tensor) -> Tensor:
        # assert z.shape == ()

        return F.relu(z)**2


class LogBarrierLoss(AbstractConstraints):
    def __init__(self, **kwargs):
        self.t: float = kwargs["t"]
        super().__init__(**kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        # assert z.shape == ()
        z_: Tensor = z.flatten()
        # del z

        barrier_part: Tensor = - torch.log(-z_) / self.t  # Careful, this part can produce NaN
        barrier_part[torch.isnan(barrier_part)] = 0
        linear_part: Tensor = self.t * z_ + -np.log(1 / (self.t**2)) / self.t + 1 / self.t
        assert barrier_part.dtype == linear_part.dtype == torch.float32

        below_threshold: Tensor = z_ <= - 1 / self.t**2
        assert below_threshold.dtype == torch.bool

        assert barrier_part.shape == linear_part.shape == below_threshold.shape
        res = barrier_part * below_threshold + linear_part * (~below_threshold)
        assert res.dtype == torch.float32

        # if z <= - 1 / self.t**2:
        #     res = - torch.log(-z) / self.t
        # else:
        #     res = self.t * z + -np.log(1 / (self.t**2)) / self.t + 1 / self.t

        assert res.requires_grad == z.requires_grad
        # print(res)

        return res


class PerpendicularLoss(nn.Module):
    # Terpstra ⊥-loss: A symmetric loss function for magnetic resonance imaging
    # reconstruction and image registration with deep learning
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def __call__(self, pred, target):
        # Assuming input of type (batch, chan, x, y)
        # Where chan is real.imaginairy
        nominator = torch.abs(pred[:, 0] * target[:, 1] - pred[:, 1] * target[:, 0])
        denominator = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)
        loss = torch.mean(nominator / (denominator + self.eps))
        return loss


class ConjugateSymmetryLoss(nn.Module):
    # Werkt tuurlijk alleen op k-space data...
    # Niet erg handig denk ik...
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, pred ,target):
        return pred - target


class TestConjugateSymmetryLoss:
    import skimage.data
    def __init__(self, kwargs):
        self.loss_obj = ConjugateSymmetryLoss(**kwargs)

    def test_noise_loss(self):
        noise_pred = torch.from_numpy(np.random.rand(5, 100, 100)).float()
        target_example = torch.from_numpy(skimage.data.astronaut()[:, :, 0] / 255.).float()


if __name__ == "__main__":
    x = np.arange(-10 * np.pi, 10 * np.pi, 0.1)
    x_sub = np.arange(-np.pi, np.pi, 0.1)
    z = np.exp(1j * x)
    phi0 = np.pi / 2
    z0 = np.exp(1j * phi0)

    plt.plot(x, np.abs(z-z0) * x)
    plt.xlim([-np.pi, np.pi])
    plt.plot(np.real(z), np.imag(z))

    A = np.random.rand(3, 2, 50, 50)
    B = A + np.random.random(A.shape)
    A_tens = torch.as_tensor(A).float()
    B_tens = torch.as_tensor(B).float()

    vgg = Vgg16(requires_grad=False)
    vgg.float()

    loss_obj = PerceptualLossStyleLoss(vgg_model=vgg)
    loss_obj(A_tens, B_tens)

    frob_loss = FrobeniusLoss()
    frob_loss(A_tens, B_tens)

    n_dmi = A_tens.ndim
    sel_axes = tuple(list(range(-(n_dmi - 1), 0)))
    frob_norm = torch.sqrt((torch.abs(A_tens) ** 2).sum(dim=(sel_axes)))

    b, d, h, w = A_tens.size()
    # reshape so we're multiplying the features for each channel
    A_tens = A_tens.view(b, d, h * w)
    A_tens.shape
    torch.einsum("bcr, brd->bcd", A_tens, A_tens.transpose(-1, 1)).shape

    # Test how we can sub sample
    import numpy as np
    import torch
    N = 100
    x_range = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x_range, x_range)
    mask = (X ** 2 + Y ** 2) < 0.25
    B = torch.from_numpy(mask)
    A = torch.from_numpy(np.random.rand(2, N,N)).float()
    C = torch.from_numpy(np.random.rand(2, N, N)).float()

    loss_rel = L1LossRelative()
    loss_rel(A, C, mask=B)

    """
    Test/debug ShimLoss
    Looks good
    """

    # Test...
    import data_generator.ShimPrediction as data_gen
    import model.ShimNet as shim_net

    ddata = '/media/bugger/MyBook/data/7T_data/b1_shim_series_resize_dataset'
    data_gen_obj = data_gen.DataGeneratorShimPrediction(ddata=ddata, dataset_type='train')
    cont = data_gen_obj.__getitem__(1)
    input_shape = (128, 128)
    model_obj = shim_net.ShimNet(in_chan=128, out_chan=16, n_downsample=3, input_shape=input_shape, debug=True)
    res = model_obj(cont['input'][None])
    loss_obj = ShimLoss()
    input = cont['input'][None]
    target = cont['target'][None]
    loss_obj(res, target, input)
    cont_loss = loss_obj.debug_call(res, target, input)
    real_input_tensor, imag_input_tensor = loss_obj.process_input(input)
    real_input_tensor.max()
    real_input_tensor.min()
    import helper.plot_class as hplotc
    hplotc.ListPlot([target, np.array(cont_loss['result'].detach()), np.array(cont_loss['target'])])
    hplotc.ListPlot()
    hplotc.SlidingPlot(cont_loss['result_interf'].detach().numpy())
    hplotc.SlidingPlot(np.array(input))

    # real_input_tensor, imag_input_tensor = self.process_input(input)

    loss_obj = AverageZero()
    loss_obj(A_tens, A_tens)

    # Test/see what the mobius transform does...

    import matplotlib.pyplot as plt
    import numpy as np

    def mobius_transf(z, a, b, c, d):
        return (a * z + b) / (c * z + d)

    phi_range = np.linspace(-10 * np.pi, 10 * np.pi, 1000)
    a = 1j * 3
    b = 20 + 1j * 6
    c = 0.001 + 1j * 2
    d = 5 + 1j * 2
    print('Mobius condition', a*d - b*c)
    res = mobius_transf(phi_range, a, b, c, d)
    plt.plot(phi_range, np.angle(res))

    # # Test storage of some files....
    a = [1,2,3,4,5,6,7,8]
    with open('/home/bugger/test_shim.txt', 'a') as f:
        f.write(', '.join([str(x) for x in a]) + '\n')
