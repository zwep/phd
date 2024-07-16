
import numpy as np
import matplotlib.pyplot as plt

"""
Het klopt dus dat alles buiten de ''grondtoon'' symmetrisch zal zijn
"""

N = 10000
N_mid = N//2
delta_x = 10
x_range = np.linspace(0, 2 * np.pi, N)
y = np.cos(x_range) + np.cos(2 * x_range) + np.cos(0.01 * x_range) + np.cos(0.001 * x_range) + np.cos(0.5 * x_range) + np.cos(5 * x_range) + np.cos(15 * x_range)
y_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(y)))
img_orig = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(y_fft)))

plt.plot(y)
plt.plot(np.abs(y_fft))

for delta_x in range(0, 2):
    y_fft_sub = np.zeros(y_fft.shape, dtype=np.complex)
    y_fft_sub[(N_mid - delta_x): (N_mid + delta_x)] = y_fft[(N_mid - delta_x): (N_mid + delta_x)]

    y_fft_sub_out = np.zeros(y_fft.shape, dtype=np.complex)
    y_fft_sub_out[:(N_mid - delta_x)] = y_fft[:(N_mid - delta_x)]
    y_fft_sub_out[(N_mid + delta_x):] = y_fft[(N_mid + delta_x):]

    img = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(y_fft_sub)))
    img_out = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(y_fft_sub_out)))
    plt.figure()
    plt.plot(img.real)
    plt.plot(img_out.real)
    plt.title(delta_x)


img_full = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(y_fft_sub + y_fft_sub_out)))

plt.plot(np.abs(y_fft_sub + y_fft_sub_out) - np.abs(y_fft))

plt.plot(img.real)
plt.plot(img_out.real)
plt.plot(y)
plt.plot((img_out + img).real)
plt.plot((img_full).real)