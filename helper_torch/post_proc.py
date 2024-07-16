
"""
Post processing functions on model output

Can be used in the training loop or outside..
"""
import helper_torch.array_transf as htarray
import torch
import math


def scale_polynomial_model(x_tens):
    return x_tens ** 2


def scale_triogonometric_model(x_tens):
    return torch.sin(x_tens)


def scale_signal_model_dummy(x_tens):
    print('FYI: There is nothing here.')
    # import numpy as np
    # x_tens = torch.as_tensor(np.random.rand(4, 2, 32, 32))
    # x_tens = x_tens.detach()
    # Input can be (batch, channel, x, y)
    # if x_tens.ndim == 4:
    #     n_batch, n_c, n_y, n_x = x_tens.shape
    # elif x_tens.ndim == 3:
    #     n_batch, n_y, n_x = x_tens.shape
    #     n_c = 1
    #     # Add a new dimension to make all the alculations the same
    #     x_tens = x_tens[:, None]
    # else:
    #     n_y = n_x = 0
    #
    # y_center, x_center = (n_y // 2, n_x // 2)
    # delta_x = int(0.1 * n_y)
    # #print('Before normalization ', x_tens.min(), x_tens.max())
    # x_tens = htarray.scale_11(x_tens, dim=(-2, -1))
    # #print('After normalization ', x_tens.min(), x_tens.max())
    # x_tens_abs = x_tens[:, 0:1] ** 2 + x_tens[:, 1:2] ** 2
    # #print('ABS After normalization ', x_tens_abs.min(), x_tens_abs.max())
    #
    # x_tens_abs_sub = x_tens_abs[:, :, y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x]
    # x_mean_abs = x_tens_abs_sub.mean(dim=(-2, -1), keepdims=True)
    #
    # x_scaled = torch.sin(x_tens_abs / x_mean_abs * math.pi / 2) ** 3
    #x_tens_angle = torch.angle(x_tens[:, 0] + 1j * x_tens[:, 1])
    # x_tens_angle = torch.atan2(x_tens[:, 0], x_tens[:, 1])
    # x_div = x_tens[:, 1] / x_tens[:, 0]
    #print('Amount of Nans obtained ', torch.isnan(x_div).sum(), x_div.shape)
    # x_div[torch.isnan(x_div)] = 0
    # x_div = htarray.scale_11(x_div, dim=(-2, -1))
    #print('xdiv min', x_div.min(), x_div.max())
    # Approximation of Tan...
    # x_tens_angle = math.pi / 4 * (x_div) + x_div * (0.186982 - 0.191942 * x_div ** 2)
    
    # x_scaled_real = x_scaled# * torch.cos(x_tens_angle*x_tens_abs)
    # x_scaled_imag = x_scaled# * torch.sin(x_tens_angle*x_tens_abs)
    # x_scaled_cpx = torch.stack([x_scaled_real, x_scaled_imag], dim=1)

    return x_tens


def scale_signal_model(x_tens):
    # x_tens = torch.as_tensor(np.random.rand(4, 2, 32, 32))
    # x_tens = x_tens.detach()
    # Input can be (batch, channel, x, y)
    if x_tens.ndim == 4:
        n_batch, n_c, n_y, n_x = x_tens.shape
    elif x_tens.ndim == 3:
        n_batch, n_y, n_x = x_tens.shape
        n_c = 1
        # Add a new dimension to make all the alculations the same
        x_tens = x_tens[:, None]
    else:
        n_y = n_x = 0

    y_center, x_center = (n_y // 2, n_x // 2)
    delta_x = int(0.1 * n_y)
    x_sub = x_tens[:, :, y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x]
    x_mean = x_sub.mean(dim=(-2, -1), keepdims=True)
    x_mean_abs = torch.sqrt(x_mean[:, 0] ** 2 + x_mean[:, 1] ** 2)
    x_tens_abs = torch.sqrt(x_tens[:, 0] ** 2 + x_tens[:, 1] ** 2)
    x_tens_angle = torch.atan2(x_tens[:, 1], x_tens[:, 0])
    # Taking the absolute values to make sure that values are between 0..1
    # B1 plus interference by complex sum. Then using abs value to scale
    x_scaled = torch.sin(x_tens_abs / x_mean_abs * math.pi / 2) ** 3
    x_scaled_real = x_scaled * torch.cos(x_tens_angle)
    x_scaled_imag = x_scaled * torch.sin(x_tens_angle)
    x_scaled_cpx = torch.stack([x_scaled_real, x_scaled_imag], dim=1)
    return x_scaled_cpx


if __name__ == "__main__":
    import data_generator.InhomogRemoval as data_gen
    dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation_rxtx'
    gen = data_gen.DataGeneratorInhomogRemoval(ddata=dir_data, dataset_type='test', complex_type='cartesian',
                                      input_shape=(1, 256, 256),
                                      alternative_input='/home/bugger/Documents/data/celeba',
                                      bias_field=True,
                                      use_tx_shim=True,
                                      b1m_scaling=False,
                                      b1p_scaling=False,
                                      debug=True,
                                      masked=True,
                                      target_type='b1p')

    container = gen.__getitem__(0)
    target_tens = container['target']
    target_scaled = scale_signal_model(target_tens[None])
    target_tens_batch = torch.stack([target_tens, target_tens, target_tens], dim=0)
    target_scaled = scale_signal_model(target_tens_batch)
    import helper.plot_fun as hplotf
    hplotf.plot_3d_list([target_scaled.numpy(), target_tens.numpy()], augm='np.real')

    # Now also test it when we are using a model output....
    import model.ResNet as model_resnet

    model_obj = model_resnet.ResnetGenerator(input_nc=16, output_nc=2)
    model_obj = model_obj.float()
    target_tens = container['target']
    with torch.no_grad():
        res = model_obj(container['input'][None])

    res_scaled = scale_signal_model(res)
    hplotf.plot_3d_list([res.numpy(), res_scaled.numpy()])
    import matplotlib.pyplot as plt
    plt.hist(res_scaled.numpy().ravel(), bins=100)

    import helper_torch.loss as hloss
    import importlib
    importlib.reload(hloss)

    spacy_loss = hloss.SpacyLoss(n_degree=5, n_size=256, transform_type='imag', debug=True)
    # loss_obj = spacy_loss(res_scaled, container['target'][None])
    loss_obj = spacy_loss(res_scaled, container['target'][None], container['mask'][None])
    _, b_list = spacy_loss.test_call(res_scaled, container['target'], container['mask'][None])

    pred_approx = [x[0].reshape(256, 256).numpy() for x in b_list]
    pred = [x[1].reshape(256, 256).numpy() for x in b_list]

    hplotf.plot_3d_list([pred_approx, pred, container['target']])

    hplotf.plot_3d_list([pred])
    hplotf.plot_3d_list([container['target']])
