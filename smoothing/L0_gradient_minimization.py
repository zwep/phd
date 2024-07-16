#!env python
# L0 gradient minimization (1D, 2D)
# License: Public Domain
#
# Implemented by
# https://github.com/t-suzuki/l0_gradient_minimization_test
# reference: "Image Smoothing via L0 Gradient Minimization", Li Xu et al., SIGGRAPH Asia 2011
# http://www.cse.cuhk.edu.hk/leojia/projects/L0smoothing/index.html

import numpy as np
from scipy.fftpack import fft, ifft, fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

def clip_img(img):
    return np.clip(img, 0, 1)

def add_noise(img, sigma):
    return  clip_img(img + np.random.randn(*img.shape) * sigma)


# 1D
def circulantshift(xs, h):
    return np.hstack([xs[h:], xs[:h]] if h > 0 else [xs[h:], xs[:h]])

def circulant_dx(xs, h):
    return (circulantshift(xs, h) - xs)

def psf2otf(psf, N):
    pad = np.zeros((N,))
    n = len(psf)
    pad[:n] = psf
    pad = np.concatenate([pad[n/2:], pad[:n/2]])
    otf = fft(pad)
    return otf

# 2D
def circulantshift2_x(xs, h):
    return np.hstack([xs[:, h:], xs[:, :h]] if h > 0 else [xs[:, h:], xs[:, :h]])

def circulantshift2_y(xs, h):
    return np.vstack([xs[h:, :], xs[:h, :]] if h > 0 else [xs[h:, :], xs[:h, :]])

def circulant2_dx(xs, h):
    return (circulantshift2_x(xs, h) - xs)

def circulant2_dy(xs, h):
    return (circulantshift2_y(xs, h) - xs)

def l0_gradient_minimization_1d(I, lmd, beta_max, beta_rate=2.0, max_iter=30, return_history=False):
    S = np.array(I).ravel()

    # prepare FFT
    F_I = fft(S)
    F_denom = np.abs(psf2otf([-1, 1], S.shape[0]))**2.0

    # optimization
    S_history = [S]
    beta = lmd*2.0
    hp = np.zeros_like(S)
    for i in range(max_iter):
        # with S, solve for hp in Eq. (12)
        hp = circulant_dx(S, 1)
        mask = hp**2.0 < lmd/beta
        hp[mask] = 0.0

        # with hp, solve for S in Eq. (8)
        S = np.real(ifft((F_I + beta*fft(circulant_dx(hp, -1))) / (1.0 + beta*F_denom)))

        # iteration step
        if return_history:
            S_history.append(np.array(S))
        beta *= beta_rate
        if beta > beta_max: break

    if return_history:
        return S_history

    return S


def l0_gradient_minimization_2d(I, lmd, beta_max, beta_rate=2.0, max_iter=30, return_history=False):
    u'''image I can be both 1ch (ndim=2) or D-ch (ndim=D)'''
    S = np.array(I)

    # prepare FFT
    F_I = fft2(S, axes=(0, 1))
    Ny, Nx = S.shape[:2]
    D = S.shape[2] if S.ndim == 3 else 1
    dx, dy = np.zeros((Ny, Nx)), np.zeros((Ny, Nx))
    dx[int(Ny/2), int(Nx/2-1):int(Nx/2+1)] = [-1, 1]
    dy[int(Ny/2-1):int(Ny/2+1), int(Nx/2)] = [-1, 1]
    F_denom = np.abs(fft2(dx))**2.0 + np.abs(fft2(dy))**2.0
    if D > 1:
        F_denom = np.dstack([F_denom]*D)

    S_history = [S]
    beta = lmd * 2.0
    hp, vp = np.zeros_like(S), np.zeros_like(S)
    for i in range(max_iter):
        # with S, solve for hp and vp in Eq. (12)
        hp, vp = circulant2_dx(S, 1), circulant2_dy(S, 1)
        if D == 1:
            mask = hp**2.0 + vp**2.0 < lmd/beta
        else:
            mask = np.sum(hp**2.0 + vp**2.0, axis=2) < lmd/beta
        hp[mask] = 0.0
        vp[mask] = 0.0

        # with hp and vp, solve for S in Eq. (8)
        hv = circulant2_dx(hp, -1) + circulant2_dy(vp, -1)
        S = np.real(ifft2((F_I + (beta*fft2(hv, axes=(0, 1))))/(1.0 + beta*F_denom), axes=(0, 1)))

        # iteration step
        if return_history:
            S_history.append(np.array(S))
        beta *= beta_rate
        if beta > beta_max: break

    if return_history:
        return S_history

    return S


if __name__ == "__main__":
    import skimage.data as data
    import helper.plot_class as hplotc
    import helper.array_transf as harray
    import helper.plot_fun as hplotf
    A = data.astronaut().mean(axis=-1)
    A = harray.scale_minmax(A)
    for lmbd in [0.01, 0.015, 0.02]:
        A_smooth = l0_gradient_minimization_2d(A, lmbd, 1.0e5)
        hplotc.ListPlot([[A, A_smooth, A - A_smooth]], title=lmbd, ax_off=True,
                        subtitle=[['original', 'smoothed', 'difference']])