"""
Check different implementations and their answers...
"""

import orthopoly.spherical_harmonic
import mpmath
import scipy.special
import numpy as np


def spherical_bessel_function(n, z):
    z = np.array(z)
    n = np.array(n)
    return np.sqrt(np.pi/(2*z)) * scipy.special.jn(n+0.5, z)


def mpmath_spherical_bessel_function(n, z):
    return np.sqrt(np.pi/(2*z)) * mpmath.besselj(n+0.5, z)


# Below we test the spherical harmonic over multiple packages..
sel_phi = np.pi/3
sel_theta = np.pi/5
sel_order = -2
sel_degree = 5
z1 = orthopoly.spherical_harmonic.sph_har(t=sel_theta, p=sel_phi, n=sel_degree, m=sel_order)
z1a = orthopoly.spherical_harmonic.sph_har_matrix(t=np.array([sel_theta]), p=np.array([sel_phi]),
                                                  yn=np.array([sel_degree]), ym=np.array([sel_order]))
z2 = scipy.special.sph_harm(sel_order, sel_degree, sel_phi, sel_theta)
z3 = mpmath.spherharm(l=sel_degree, m=sel_order, theta=sel_theta, phi=sel_phi)
print(f'{z1}, {z2}, {z3}')

# Below we test the spherical bessel function
scipy.special.spherical_jn(n=-2, z=1)
res_scipy = spherical_bessel_function(3, np.arange(-2, 10))
res_mp = [mpmath_spherical_bessel_function(3, x).__float__() for x in np.arange(0, 10)]
plt.plot(np.arange(-2, 10), res_scipy)
plt.plot(np.arange(0, 10), res_mp)

scipy.special.jn(2, 1)
mpmath.besselj(n=2, z=1)



"""
Test here (spherical) bessel functions..?
"""
n_degree = 0.5
x_pos = 2
# Integer Bessel function.. are called cylindrical functions
z = mpmath.besselj(n_degree, x_pos, derivative=0)
tau_range = np.linspace(0, np.pi, 1000)
if n_degree.is_integer():
    z1 = (1/np.pi) * sum(np.cos(n_degree * tau_range - x_pos * np.sin(tau_range))) * np.diff(tau_range)[0]
    print(z, z1)
else:
    t_range = np.linspace(0, 120, 1000)
    z_integer = (1 / np.pi) * sum(np.cos(n_degree * tau_range - x_pos * np.sin(tau_range))) * np.diff(tau_range)[0]
    z_non_integer = z_integer - np.sin(n_degree * np.pi) / np.pi * sum(np.exp(-x_pos * np.sinh(t_range) - n_degree * t_range)) * np.diff(t_range)[0]
    print(z, z_non_integer)

# Non-integer Bessel function.. are called sperical functions
mpmath.besselj(-10.5, 5, derivative=0)
