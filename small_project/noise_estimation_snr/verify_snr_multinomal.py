import numpy as np
import math
import matplotlib.pyplot as plt
import helper.array_transf as harray
import helper.misc as hmisc
import math

"""
The idea here is that we start with multi multivariate normal called Z_i. With this we create complex noise for
each coil. Then we take the sum of absolutes, or sum of squares of this for all coils. In summary

Z_{i, 1}, Z_{i, 2} ~ N(0, mu)
Y_i = Z_{i, 1} + 1j * Z_{i, 2}
X_1 = \sum_{i=1}^n |Y_i| =  \sum_{i=1}^n \sqrt{Z_{i, 1}^2 +  Z_{i, 2}^2} = \sqrt{2} \sum_{i=1}^n |Z_i|
X_2 = \sum_{i=1}^n |Y_i| ^ 2 = \sum_{i=1}^n Z_{i, 1}^2 +  Z_{i, 2}^2 = 2 \sum_{i=1}^n Z_{i}^2 

Then we:
- Do a sum of squares (X_1) --> This results in a sum of folded normal distributions, because the mean is zero
- Do a sum of absolutes (X_2) --> This results in a central chi squared distribution, because the mean is zero

In both cases we want to have an expression for the variance.

"""

n_coils = 8
n_samples = int(1e5)
mu_noise = 0
SNR = 25
SNR_range = np.arange(25, 40)
# var_noise = 1 / (2 * SNR * np.sqrt(n_coils))
var_noise = (2 / (4-np.pi)) * (1 / (SNR ** 2 * (n_coils)))
var_noise_list = (2 / (4-np.pi)) * (1 / (SNR_range ** 2 * (n_coils)))
bool_array = np.array([x * 100 for x in var_noise_list]) < 0.05
SNR_range[bool_array]
var_noise_list[bool_array]

std_noise = np.sqrt(var_noise)

Z_1 = np.random.multivariate_normal(np.ones(n_coils) * mu_noise,
                                    np.eye(n_coils) * var_noise,
                                    size=n_samples)
Z_2 = np.random.multivariate_normal(np.ones(n_coils) * mu_noise,
                                    np.eye(n_coils) * var_noise,
                                    size=n_samples)
Y = (Z_1 + 1j * Z_2)
X_1 = np.abs(Y).sum(axis=-1)
import matplotlib.pyplot as plt
plt.imshow(X_1.reshape(hmisc.get_square(n_samples)))

X_2 = (np.abs(Y) ** 2).sum(axis=-1)

# Verification of X_1
# Source: https://en.wikipedia.org/wiki/Rayleigh_distribution
analytic_mean_X_1 = n_coils * std_noise * np.sqrt(np.pi / 2)
analytic_var_X_1 = n_coils * (4 - np.pi) / 2 * var_noise

calculated_mean_X_1 = np.mean(X_1)
calculated_var_X_1 = np.var(X_1)

print(f'\n1 / sigma of X_1 {(1 / np.sqrt(calculated_var_X_1)):.3f}')

# Verification of X_2
analytic_mean_X_2 = 2 * n_coils * (mu_noise ** 2 + std_noise ** 2)
analytic_var_X_2 = 2 * n_coils * (4 * (std_noise ** 2) * (mu_noise ** 2) + 2 * std_noise ** 4)

calculated_mean_X_2 = np.mean(X_2)
calculated_var_X_2 = np.var(X_2)

print(f'\n1 / sigma of X_2 {(1 / np.sqrt(calculated_var_X_2)):.3f}')


