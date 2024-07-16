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

The code below is only valid if the assumed expectation is zero
"""

n_coils = 1
n_samples = int(1e5)
mu_noise = 0
var_noise = 3
std_noise = np.sqrt(var_noise)


Z_1 = np.random.multivariate_normal(np.ones(n_coils) * mu_noise,
                                    np.eye(n_coils) * var_noise,
                                    size=n_samples)
Z_2 = np.random.multivariate_normal(np.ones(n_coils) * mu_noise,
                                    np.eye(n_coils) * var_noise,
                                    size=n_samples)
Y = (Z_1 + 1j * Z_2)
X_1 = np.abs(Y)
X_2 = np.abs(Y) ** 2

# Verification of X_1
# Source: https://en.wikipedia.org/wiki/Rayleigh_distribution
analytic_mean_X_1 = std_noise * np.sqrt(np.pi / 2)
analytic_var_X_1 = (4 - np.pi) / 2 * var_noise

calculated_mean_X_1 = np.mean(X_1)
calculated_var_X_1 = np.var(X_1)

print(f'Variance of X_1 {calculated_var_X_1:.3f} -*- {analytic_var_X_1:.3f}')
print(f'Mean of X_1     {calculated_mean_X_1:.3f} -*- {analytic_mean_X_1:.3f}')


# Verification of X_2
# Source: https://math.stackexchange.com/questions/2506784/deriving-variance-of-non-central-chi-square/2506832?noredirect=1#comment9833605_2506832
analytic_mean_X_2 = 2 * (mu_noise ** 2 + std_noise ** 2)
analytic_var_X_2 = 2 * (4 * n_coils * (std_noise ** 2) * (mu_noise ** 2) + 2 * n_coils *std_noise ** 4)

calculated_mean_X_2 = np.mean(X_2)
calculated_var_X_2 = np.var(X_2)

print(f'\nVariance of X_2 {calculated_var_X_2:.3f} -*- {analytic_var_X_2:.3f}')
print(f'Mean of X_2     {calculated_mean_X_2:.3f} -*- {analytic_mean_X_2:.3f}')


"""
For later:

Folded Normal distribution

mean
variance : sigma ^2 + mu ^2 

Non-centered chi squared distribtuion (sum of squares, mean mu and unit variance, k degrees of freedom)

mean: k + lambda
variance: 2 * (k + 2 * lambda)   
lambda = sum of squared means


Chi squared (sum of squares of a standard normal distributions, with L degrees of freedom) 

mean: L
variance: 2L

Rayleigh (complex absolute sum with real and imaginairy parts normal distributed (mean zero, equal variance)

mean: sigma * np.sqrt(np.pi / 2)
variance: (4-np.pi)/2* sigma ** 2

Rice (the probability distribution of the magnitude of a circularly-symmetric bivariate normal random variable, possibly with non-zero mean (noncentral))

Mean: nasty
variance: nasty..

Gaussian / normal distribution

mean: mu
variance: sigma ** 2

"""