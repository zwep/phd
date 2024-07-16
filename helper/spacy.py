import matplotlib.pylab as plt
import numpy as np
import torch
import scipy.special
import orthopoly
import torch.nn.functional as F


def spherical_bessel_function(n, z):
    z = np.array(z)
    n = np.array(n)
    return np.sqrt(np.pi/(2*z)) * scipy.special.jn(n+0.5, z)


def get_spacy_matrix(n_x, n_y=None, n_degree=20, offset_x=0, offset_y=0, x_range=(-1, 1), y_range=(-1, 1),
                     debug=False, epsilon_r=50, sigma=0.6):
    if n_y is None:
        n_y = n_x

    x_0, x_1 = x_range
    y_0, y_1 = y_range

    x_range = np.linspace(x_0, x_1, n_x) + offset_x
    y_range = np.linspace(y_0, y_1, n_y) + offset_y
    X, Y = np.meshgrid(x_range, y_range)
    r = np.sqrt(X ** 2 + Y ** 2)
    # r = r / np.max(r)
    # Theta: Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.
    theta = np.zeros(r.shape)
    # Phi : Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
    phi = np.arctan2(Y, X)

    theta[phi < 0] = np.pi
    phi = np.abs(phi)

    epsilon_0 = 8.8 * 1e-12
    mu_0 = 4 * np.pi * 1e-7
    omega = 42.58 * 7
    complex_wave_number = np.sqrt(mu_0 * epsilon_0 * epsilon_r * omega ** 2 + 1j * omega * sigma)

    if debug:
        print('R ', r.min(), r.mean(), r.max())
        print('Phi ', phi.min(), phi.mean(), phi.max())
        print('Theta ', theta.min(), theta.mean(), theta.max())

    res = []
    # Shherical part...
    for i_degree in range(n_degree):
        jn = scipy.special.spherical_jn(n=i_degree, z=complex_wave_number * r)
        for m_order in range(-i_degree, i_degree):
            # Spherical part...
            temp_sph = jn * orthopoly.spherical_harmonic.sph_har(t=phi, p=theta, n=i_degree, m=m_order)
            res.append(temp_sph)

    for i_degree in range(-n_degree, n_degree):
        # Cylindrical part - based on Mikes expansion
        temp_cyl = scipy.special.jn(i_degree, complex_wave_number * r) * np.exp(1j * i_degree * phi)
        res.append(temp_cyl)

    # This is a lot easier
    n_equations = len(res)
    if debug:
        print('Amount of equations combined ', n_equations)
    spacy_matrix = np.array(res).reshape(n_equations, -1).T

    if debug:
        print('Shape of NaNs in spacy ', spacy_matrix[np.isnan(spacy_matrix)].shape)
        return spacy_matrix, {'r': r, 'phi': phi, 'theta': theta}
    else:
        return spacy_matrix


def get_cylindrical_matrix(n_x, n_y=None, n_order=20, offset_x=0, offset_y=0, x_range=(-1, 1), y_range=(-1, 1),
                          debug=False, epsilon_r=50, sigma=0.6):
    if n_y is None:
        n_y = n_x

    x_0, x_1 = x_range
    y_0, y_1 = y_range

    x_range = np.linspace(x_0, x_1, n_x) + offset_x
    y_range = np.linspace(y_0, y_1, n_y) + offset_y

    epsilon_0 = 8.8 * 1e-12
    mu_0 = 4 * np.pi * 1e-7
    omega = 42.58 * 7
    wave_number = np.sqrt(mu_0 * epsilon_0 * epsilon_r * omega ** 2 + 1j * omega * sigma)

    X, Y = np.meshgrid(x_range, y_range)
    r = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    res = []
    for i_order in range(-n_order, n_order):
        # Cylindrical part - based on Mikes expansion
        temp_cyl = scipy.special.jn(i_order, wave_number * r) * np.exp(1j * i_order * theta)
        res.append(temp_cyl)
        temp_cyl = scipy.special.jn(i_order, wave_number * r) * np.exp(-1j * i_order * theta)
        res.append(temp_cyl)
        # These below are trouble makes because of singularity in the origin
        # temp_cyl = scipy.special.yn(i_order, wave_number.real * r) * np.exp(-1j * i_order * theta)
        # res.append(temp_cyl)
        # temp_cyl = scipy.special.yn(i_order, wave_number.real * r) * np.exp(1j * i_order * theta)
        # res.append(temp_cyl)

    n_equations = len(res)
    if debug:
        print('Amount of equations combined ', n_equations)
    spacy_matrix = np.array(res).reshape(n_equations, -1).T

    return spacy_matrix


def get_spherical_matrix(n_x, n_y=None, n_degree=2, offset_x=0, offset_y=0, x_range=(-1, 1), y_range=(-1, 1),
                         debug=False, epsilon_r=50, sigma=0.6):
    # Please mind that the interpretation of `order` and `degree` is different from the cylindrical
    # In cylindrical we talk about order of the Bessel function
    # In spherical we talk about order of the Legendre Polynomial. Then the degree denotes the order of the Bessel function

    if n_y is None:
        n_y = n_x

    x_0, x_1 = x_range
    y_0, y_1 = y_range

    x_range = np.linspace(x_0, x_1, n_x) + offset_x
    y_range = np.linspace(y_0, y_1, n_y) + offset_y
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros((n_x, n_y))
    r = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)
    theta_cos = np.cos(theta)
    phi = np.arctan2(Z, X)

    epsilon_0 = 8.8 * 1e-12
    mu_0 = 4 * np.pi * 1e-7
    omega = 42.58 * 7
    wave_number = np.sqrt(mu_0 * epsilon_0 * epsilon_r * omega ** 2 + 1j * omega * sigma)

    leg_pol = [scipy.special.lpmn(m=n_degree, n=n_degree, z=x)[0] for x in theta_cos.ravel()]
    leg_pol = np.array(leg_pol).reshape((n_y, n_x, n_degree+1, n_degree+1))

    res = []
    for i_degree in range(n_degree):
        sph_bessel = scipy.special.jn(i_degree + 1 / 2, wave_number * r)
        R_factor = sph_bessel / (wave_number * r) ** 0.5
        for i_order in range(0, i_degree):
            PHI_factor = np.exp(-1j * i_order * phi)
            THETA_factor = leg_pol[:, :, i_order, i_degree]
            sph_bessel = R_factor * PHI_factor * THETA_factor
            res.append(sph_bessel)

            # PHI_factor = np.exp(1j * i_order * phi)
            # THETA_factor = leg_pol[:, :, i_order, i_degree]
            # sph_bessel = R_factor * PHI_factor * THETA_factor
            # res.append(sph_bessel)

    n_equations = len(res)

    if debug:
        print('Amount of equations combined ', n_equations)
    spacy_matrix = np.array(res).reshape(n_equations, -1).T

    return spacy_matrix


def spacy_approximation(spacy_matrix, target, mask=None, spacy_inv=None):
    # Target should be in (-1, ) shape
    # Mask should be in (-1, 1) shape..
    if spacy_inv is None:
        spacy_inv = np.linalg.pinv(spacy_matrix)

    if mask is not None:
        mask_ind = mask[:, 0] == 1
        spacy_mask_inv = spacy_inv[:, mask_ind]
        spacy_mask = spacy_matrix[mask_ind]
        target_mask = target[mask_ind]

        x_approx_mask = np.matmul(spacy_mask_inv, target_mask)
        target_approx_mask = np.matmul(spacy_mask, x_approx_mask)

        target_approx = np.zeros(target.shape, dtype=complex)
        target_approx[mask_ind] = target_approx_mask
    else:
        x_approx = np.matmul(spacy_inv, target)
        target_approx = np.matmul(spacy_matrix, x_approx)

    return target_approx


def spacy_approximation_torch(spacy_tensor, target_tensor, mask=None, spacy_inv=None):
    # Assumes input to be the shape of (-1, n_batch)

    # If we dont have an inverse.. calculate it
    if spacy_inv is None:
        spacy_inv = torch.pinverse(spacy_tensor)

    if mask is not None:
        mask_ind = mask[:, 0] == 1
        spacy_mask_inv = spacy_inv[:, mask_ind]
        spacy_mask = spacy_tensor[mask_ind]
        target_tensor_mask = target_tensor[mask_ind]

        x_approx_mask = torch.matmul(spacy_mask_inv, target_tensor_mask)
        target_approx_mask = torch.matmul(spacy_mask, x_approx_mask)

        target_approx = torch.zeros(target_tensor.shape)
        target_approx[mask_ind] = target_approx_mask
    else:
        x_pred_tensor = torch.matmul(spacy_inv, target_tensor)
        target_approx = torch.matmul(spacy_tensor, x_pred_tensor)

    return target_approx


def check_helmholtz(x, kernel="6", dx=1, wave_number=1):
    kernel_2 = np.array([1, -2, 1])
    kernel_4 = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])
    kernel_6 = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
    kernel_8 = np.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560])
    kernel_dict = {'2': [kernel_2, 0], '4': [kernel_4, 1], '6': [kernel_6, 2], '8': [kernel_8, 3]}

    kernel, pad = kernel_dict[kernel]
    n_kernel = len(kernel)

    kernel_xx = np.tile(kernel, n_kernel).reshape(n_kernel, -1)
    kernel_xx = torch.nn.Parameter(torch.from_numpy(kernel_xx).view(1, 1, n_kernel, n_kernel).float(), requires_grad=False)
    filter_xx = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1, bias=False)
    filter_xx.weight.data = kernel_xx
    filter_xx.weight.requires_grad = False
    filter_xx.kernel_size = n_kernel

    kernel_yy = np.tile(kernel.T, n_kernel).reshape(n_kernel, -1)
    kernel_yy = torch.nn.Parameter(torch.from_numpy(kernel_yy).view(1, 1, n_kernel, n_kernel).float(), requires_grad=False)
    filter_yy = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1, bias=False)
    # Transposed here takes care of errthing
    filter_yy.weight.data = kernel_yy
    filter_yy.weight.requires_grad = False
    filter_yy.kernel_size = n_kernel

    operator_xx = F.pad(filter_xx(x), (pad, pad, pad, pad))
    operator_yy = F.pad(filter_yy(x), (pad, pad, pad, pad))

    helmholtz_pred = operator_xx + operator_yy + dx ** 2 * wave_number * x
    return helmholtz_pred


def check_spacy(x, mask, n_degree=8):
    A_spacy = get_spacy_matrix(*x.shape[::-1], n_degree=n_degree)
    A_spacy_pinv = np.linalg.pinv(A_spacy)
    y_pred_ravel_mask = x.reshape(-1,)
    mask_ravel = mask.reshape(-1)
    y_pred_ravel_mask = y_pred_ravel_mask[mask_ravel == 1]

    spacy_mask_inv = A_spacy_pinv[:, mask_ravel == 1]
    spacy_mask = A_spacy[mask_ravel == 1]
    x_pred_tensor = np.matmul(spacy_mask_inv, y_pred_ravel_mask)
    b_pred_masked = np.matmul(spacy_mask, x_pred_tensor)
    b = np.zeros(x.shape, dtype='complex').reshape(-1)
    b[mask_ravel == 1] = b_pred_masked
    b = b.reshape(x.shape)
    return b

"""
old part of the code....
    # Shherical part...
    for i_degree in range(n_degree):
        for m_order in range(0, i_degree):
            temp = scipy.special.spherical_jn(n=m_order, z=complex_wave_number*r) * scipy.special.sph_harm(m_order, i_degree, phi, theta)
            res.append(temp)

    # Cylindrical part
    for i_degree in range(n_degree):
        for m_order in range(-i_degree, i_degree):
            temp = scipy.special.jv(m_order, complex_wave_number*r) * scipy.special.sph_harm(m_order, i_degree, phi, theta)
            res.append(temp)
"""