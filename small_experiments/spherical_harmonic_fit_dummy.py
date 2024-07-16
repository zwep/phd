

import helper.plot_fun as hplotf
import scipy.special
import numpy as np
import matplotlib.pyplot as plt


def spherical_bessel_function(n, z):
    z = np.array(z)
    n = np.array(n)
    return np.sqrt(np.pi/(2*z)) * scipy.special.jn(n+0.5, z)

# Create a dummy domain
n = 100
N = n ** 2
x_range = np.linspace(-1, 1, n) + 5
y_range = np.linspace(-1, 1, n) + 5
X, Y = np.meshgrid(x_range, y_range)
Z = np.cos(2*np.pi * X) * np.cos(2*np.pi * Y)
b = Z.reshape(-1)

n_degree = 50

# Create r, theta and phi..
r = np.sqrt(X ** 2 + Y ** 2)
theta = np.arctan2(Y, X)
phi = np.zeros(theta.shape)
phi[theta < 0] = np.pi
theta = np.abs(theta)

hplotf.plot_3d_list([r, theta, phi])

# % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # %
# % # % # % # %     Cylindrical         # % # % # % # % # % # %
# % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # %

"""Cylindrical only
Mikes implementation"""

# Now create the cylindrical matrix
res = []
for m_order in range(-n_degree, n_degree):
    temp = scipy.special.jv(m_order, r) * np.exp(1j * m_order * theta)
    res.append(temp)
A_mike = np.array(res).reshape(n_degree * 2, -1).T

"""Cylindrical only
Janots implementation"""

res = []
for i_degree in range(n_degree):
    for m_order in range(0, i_degree):
        temp = scipy.special.jv(m_order, r) * scipy.special.sph_harm(m_order, i_degree, phi, theta)
        res.append(temp)

A_janot = np.array(res).reshape((n_degree**2-n_degree)//2, -1).T

"""Cylindrical only
My implementation"""

res = []
for i_degree in range(n_degree):
    for m_order in range(-i_degree, i_degree):
        temp = scipy.special.jv(m_order, r+2) * scipy.special.sph_harm(m_order, i_degree, phi, theta)
        res.append(temp)

A_seb = np.array(res).reshape((n_degree**2-n_degree), -1).T

for A in [('mike cylindrical', A_mike), ('janot cylindrical', A_janot), ('janot cylindrical', A_seb)]:
    proposed_x = np.linalg.lstsq(A, b, rcond=None)
    proposed_x = proposed_x[0]

    plt.figure()
    # Plot as single line
    plt.plot(np.matmul(A, proposed_x).real, '.-', c='r', alpha=0.2)
    plt.plot(b.real, '.-', c='k', alpha=0.2)

    # Reshape to image
    pred_1 = np.matmul(A, proposed_x).reshape(n, n)
    true_0 = b.reshape(n, n)
    hplotf.plot_3d_list([pred_1, true_0], augm='np.real', subtitle=[['pred'], ['true']])


# % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # %
# % # % # % # %     Spherical           # % # % # % # % # % # %
# % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # %

""" Spherical component only
with negative morder """

# Create Spherical components
res = []

for i_degree in range(n_degree):
    # Need to have order larger than 0 because spherical_jn cannot handle it..
    # Janot, Mike and Alessandro all deal with negative order... Dont know how..
    for m_order in range(-i_degree, i_degree):
        temp = spherical_bessel_function(n=m_order, z=r) * scipy.special.sph_harm(m_order, i_degree, phi, theta)
        res.append(temp)

n_equations = (n_degree**2-n_degree)
A_negative = np.array(res).reshape(n_equations, -1).T


""" Spherical component only
without negative morder """

# Create Spherical components
res = []

for i_degree in range(n_degree):
    for m_order in range(0, i_degree):
        temp = scipy.special.spherical_jn(n=m_order, z=r) * scipy.special.sph_harm(m_order, i_degree, phi, theta)
        res.append(temp)

n_equations = (n_degree**2-n_degree)//2
A_positive = np.array(res).reshape(n_equations, -1).T


for i_title, A in [('spherical positive', A_positive), ('spherical negative', A_negative)]:
    proposed_x = np.linalg.lstsq(A, b, rcond=None)
    proposed_x = proposed_x[0]

    # Reshape to image
    pred_1 = np.matmul(A, proposed_x).reshape(n, n)
    true_0 = b.reshape(n, n)
    hplotf.plot_3d_list([pred_1, true_0], augm='np.real', subtitle=[['pred'], ['true']], title=i_title)

# % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # %
# % # % # % # %     Spherical and Cylindrical % # % # % # % # %
# % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # %

""" Spherical and cylindrical functions together.. """

# Create Spherical components
res = []
# Too high degree causes NaNs
n_degree = 25

for i_degree in range(n_degree):
    # Zonder de negatieve termen gaat het zo veel beter.
    for m_order in range(0, i_degree):
        temp = spherical_bessel_function(n=m_order, z=r) * scipy.special.sph_harm(m_order, i_degree, phi, theta)
        res.append(temp)

for i_degree in range(n_degree):
    for m_order in range(-i_degree, i_degree):
        temp = scipy.special.jv(m_order, r) * scipy.special.sph_harm(m_order, i_degree, phi, theta)
        res.append(temp)

n_equations = (n_degree**2-n_degree)//2 + (n_degree**2-n_degree)
A_positive = np.array(res).reshape(n_equations, -1).T


# Create Spherical components
res = []
# Too high degree causes NaNs
n_degree = 25

for i_degree in range(n_degree):
    # Zonder de negatieve termen gaat het zo veel beter.
    for m_order in range(-i_degree, i_degree):
        temp = spherical_bessel_function(n=m_order, z=r) * scipy.special.sph_harm(m_order, i_degree, phi, theta)
        res.append(temp)

for i_degree in range(n_degree):
    for m_order in range(-i_degree, i_degree):
        temp = scipy.special.jv(m_order, r) * scipy.special.sph_harm(m_order, i_degree, phi, theta)
        res.append(temp)

n_equations = (n_degree**2-n_degree) + (n_degree**2-n_degree)
A_negative = np.array(res).reshape(n_equations, -1).T

#####

b = Z.reshape(-1)

for i_title, A in [('spherical/cylindrical positive', A_positive), ('spherical/cylindrical negative', A_negative)]:
    proposed_x = np.linalg.lstsq(A, b, rcond=None)
    proposed_x = proposed_x[0]

    # Reshape to image
    pred_1 = np.matmul(A, proposed_x).reshape(n, n)
    true_0 = b.reshape(n, n)
    hplotf.plot_3d_list([pred_1, true_0], augm='np.real', subtitle=[['pred'], ['true']], title=i_title)
