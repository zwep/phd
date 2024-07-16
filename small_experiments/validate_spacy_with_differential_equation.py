
"""
CHeck if the Bessel functions created by numpy really satisfy the Bessel equation

SImilar for the basis functions and the Helmholtz equation
"""

import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import orthopoly
import numpy as np
import scipy.special
import os
import helper.spacy as hspacy
import matplotlib.pyplot as plt


def get_first_derivative(x, dx, axis=0):
	return np.gradient(x, axis=axis) * (1/dx)


def get_second_derivative(x, dx, axis=0):
	return np.gradient(np.gradient(x, axis=axis) * (1/dx), axis=axis) * (1/dx)


"""
Check for the Laplace equation
"""
n = 50
x_range = np.linspace(-2, 2, n)
dx_factor = (1 / np.diff(x_range)[0])
y = x_range ** 2
laplace_solution = 2 * np.ones(n)
y_x = np.gradient(y, axis=0)* dx_factor
y_xx = np.gradient(y_x, axis=0) * dx_factor

plt.figure()
plt.plot(y_xx, 'r', label='numeric solution')
plt.plot(laplace_solution, 'k', label='solution')
plt.legend()

"""
Check for the Bessel Equation
"""


def check_bessel_solution_np(order, x_range):
	# Scale factor is used when checking Helmholtz equation derivation
	dx = np.diff(x_range)[0]
	bessel = scipy.special.jn(order, x_range)
	bessel_x = np.gradient(bessel, axis=0) * (1/dx)
	bessel_xx = np.gradient(bessel_x, axis=0) * (1/dx)
	bessel_eq = x_range ** 2 * bessel_xx + x_range * bessel_x + (x_range ** 2 - order ** 2) * bessel
	return bessel_eq


def check_bessel_solution_jvp(order, x_range):
	# The usage of jvp is far superior to the gradient solution
	# Scale factor is used when checking Helmholtz equation
	bessel = scipy.special.jn(order, x_range)
	bessel_x = scipy.special.jvp(order, x_range, n=1)
	bessel_xx = scipy.special.jvp(order, x_range, n=2)

	bessel_eq = x_range ** 2 * bessel_xx + x_range * bessel_x + (x_range ** 2 - order ** 2) * bessel
	return bessel_eq


def check_bessel_solution_yvp(order, x_range):
	# The usage of jvp is far superior to the gradient solution
	# Scale factor is used when checking Helmholtz equation
	bessel = scipy.special.yn(order, x_range)
	bessel_x = scipy.special.yvp(order, x_range, n=1)
	bessel_xx = scipy.special.yvp(order, x_range, n=2)

	bessel_eq = x_range ** 2 * bessel_xx + x_range * bessel_x + (x_range ** 2 - order ** 2) * bessel
	return bessel_eq

x_range = np.linspace(-2, 2, 100)
delta_x = np.diff(x_range)[0]
for i_order in range(10):
	bessel = scipy.special.jn(i_order, x_range)
	bessel_eq = check_bessel_solution_jvp(i_order, x_range)
	bessel_eq_np = check_bessel_solution_np(i_order, x_range)
	plt.figure()
	plt.plot(x_range, bessel_eq, 'r')
	plt.plot(x_range, bessel_eq_np, 'k')
	plt.title(str(i_order))
	plt.ylim(-0.01, 0.01)

# Check a modified version of the Bessel equation

x_range = np.linspace(0, 2, 100)
delta_x = np.diff(x_range)[0]
n = 0
for k in range(1, 10):
	plt.figure()
	for i_order in range(10):
		bessel_eq_np = check_bessel_solution_jvp(i_order, x_range * k)
		plt.plot(bessel_eq_np)
		plt.ylim(-0.0001, 0.0001)

"""
Check for a simple second order problem
"""

def check_second_order_ode_cpx(wave_number, theta_range):
	cylindrical_factor = np.exp(wave_number * theta_range * 1j)
	cyl_x = np.gradient(cylindrical_factor, axis=0) * (1 / delta_theta)
	cyl_xx = np.gradient(cyl_x, axis=0) * (1 / delta_theta)
	solution = (1 / cylindrical_factor) * cyl_xx
	return solution


def check_second_order_ode(wave_number, theta_range):
	cylindrical_factor = np.exp(wave_number * theta_range)
	cyl_x = np.gradient(cylindrical_factor, axis=0) * (1 / delta_theta)
	cyl_xx = np.gradient(cyl_x, axis=0) * (1 / delta_theta)
	solution = (1 / cylindrical_factor) * cyl_xx
	return solution

n = 1000
theta_range = np.linspace(0, 2*np.pi, n)
delta_theta = np.diff(theta_range)[0]

for wave_number in range(10):
	second_order_solution = check_second_order_ode_cpx(wave_number, theta_range)
	plt.figure()
	plt.plot(second_order_solution.real)
	plt.plot(-wave_number ** 2 * np.ones(n), 'k--')

n = 1000
z_range = np.linspace(-2, 2, n)
delta_z = np.diff(z_range)[0]

for wave_number in range(10):
	second_order_solution = check_second_order_ode(wave_number, theta_range)
	plt.figure()
	plt.plot(second_order_solution.real)
	plt.plot(wave_number ** 2 * np.ones(n), 'k--')

"""
Check for the Helmholtz Equation (cylindrical coordinates)
"""

n = 10000
r_range = np.linspace(0, 1, n)
delta_r = np.diff(r_range)[0]

theta_range = np.linspace(0, 2*np.pi, n)
delta_theta = np.diff(theta_range)[0]

z_range = np.linspace(-0.01, 0.01, n)
delta_z = np.diff(z_range)[0]

plot_helmholtz_factors = False
plot_helmholtz_derivative = False

# Check
wave_number = np.pi
z_seperation = 5
max_order = 3
for z_seperation in range(-3, 3):
	for order in range(-3, 3):
		r_scale_factor = np.sqrt(z_seperation ** 2 + wave_number ** 2)
		s_range = r_range * r_scale_factor
		delta_s = np.diff(s_range)[0]
		R_factor = scipy.special.jn(order, s_range)
		R_x = scipy.special.jvp(order, s_range, n=1)
		R_xx = scipy.special.jvp(order, s_range, n=2)

		THETA_factor = np.exp(-order * theta_range * 1j)
		THETA_xx = get_second_derivative(THETA_factor, delta_theta)
		# plt.plot((1/THETA_factor) * THETA_xx)
		# plt.plot(-order ** 2* np.ones(n))

		# z_factor = np.exp(z_seperation * z_range)
		Z_factor = np.exp(-z_seperation * z_range)
		Z_xx = get_second_derivative(Z_factor, delta_z)
		# plt.plot((1/Z_factor) * Z_xx)
		# plt.plot(z_seperation ** 2 * np.ones(n))

		# Multiplied by r
		helmholtz_eq1 = \
						s_range ** 2 * R_xx + s_range * R_x + \
						(R_factor/THETA_factor) * THETA_xx + \
						(R_factor * r_range ** 2/Z_factor) * Z_xx + \
						R_factor * r_range ** 2 * wave_number ** 2

		plt.figure()
		plt.plot(helmholtz_eq1.real)
		plt.title(f'order {order} degree {z_seperation}')


"""
Check for the spherical helmholtz equation solver

This part is about the spherical bessel function
"""

n = 10000
x_range = np.linspace(0, 1, n)
delta_x = np.diff(x_range)[0]
order = 3
k = 12
sph_bessel = scipy.special.jn(order+1/2, k * x_range)
# This is the substitution I used.. but the one below also works...
# sph_bessel = sph_bessel / (k * x_range) ** 0.5
sph_bessel = np.pi ** 0.5 * sph_bessel / (2 * x_range) ** 0.5
sph_bessel_x = (1/delta_x) * np.gradient(sph_bessel, axis=0)
sph_bessel_xx = (1/delta_x) * np.gradient(sph_bessel_x, axis=0)

test_eq = x_range ** 2 * sph_bessel_xx + \
		  2 * x_range * sph_bessel_x + \
		  (k ** 2 * x_range ** 2 - order * (order + 1)) * sph_bessel

test_eq[np.isnan(test_eq)] = 0
plt.plot(test_eq[:-2])
plt.ylim(-0.01, 0.01)

"""
Check associated Legendre Polynomials
"""

n = 100000
x_range = np.linspace(-1, 1, n)
delta_x = np.diff(x_range)[0]

order = 5
l = 5
leg_pol = [scipy.special.lpmn(m=order, n=l, z=x)[0] for x in x_range]
leg_pol = np.array(leg_pol)

# http://physics.ucsc.edu/~peter/116C/helm_sp.pdf

for l_sel in range(l):
	for m_sel in range(-l_sel, l_sel):
		# m_sel = 1
		# l_sel = 2
		legendre = leg_pol[:, m_sel, l_sel]

		legendre_x = np.gradient(legendre, axis=0) * (1/delta_x)
		sub_part = (1 - x_range ** 2) * legendre_x
		sub_part_x = np.gradient(sub_part, axis=0) * (1/delta_x)

		test_eq = sub_part_x + (l_sel * (l_sel+1) - m_sel ** 2 / (1 - x_range ** 2)) * legendre
		plt.figure()
		plt.plot(x_range, test_eq)
		# plt.ylim(-0.0001, 0.0001)#
		plt.title(f'order {m_sel} degree {l_sel}')

"""
Now check substituted with x=cos(theta)
"""

n = 100000
# theta_range = np.linspace(-np.pi, np.pi, n)
theta_range = np.linspace(0, 2*np.pi, n)
delta_theta = np.diff(theta_range)[0]
x_range = np.cos(theta_range)

order = 5
l = 7
# http://physics.ucsc.edu/~peter/116C/helm_sp.pdf

leg_pol = [scipy.special.lpmn(m=order, n=l, z=x)[0] for x in x_range]
leg_pol = np.array(leg_pol)
for l_sel in range(l):
	for m_sel in range(l_sel):
# m_sel = 5
# l_sel = 12
		legendre = leg_pol[:, m_sel, l_sel]
		plot_intermediate = False

		# analytical shape of Legendre POlynomial in (cos(theta))
		analytic_legendre = 3 * (1 - x_range ** 2)
		if plot_intermediate:
			plt.plot(analytic_legendre, 'b*-', alpha=0.5)
			plt.plot(legendre, 'r.-', alpha=0.5)

		# analytical shape of Legendre POlynomial in d/dtheta (cos(theta))
		analytic_legendre_theta = 6 * np.sin(theta_range) * np.cos(theta_range)
		legendre_x = np.gradient(legendre, axis=0) * (1/delta_theta)
		if plot_intermediate:
			plt.plot(analytic_legendre_theta, 'b*-', alpha=0.5)
			plt.plot(legendre_x, 'r.-', alpha=0.5)

		# analytical shape of Legendre POlynomial in (1/sintheta) d/dtheta (sintheta d/dtheta (cos(theta)))
		analytic_legendre_theta_theta = 2 * 6 * np.cos(theta_range) ** 2 - 6 * np.sin(theta_range) ** 2
		sub_part = np.sin(theta_range) * legendre_x
		sub_part_x = np.gradient(sub_part, axis=0) * (1/delta_theta) * np.sin(theta_range) # * (1 / np.sin(theta_range))
		if plot_intermediate:
			plt.plot(analytic_legendre_theta_theta, 'b*-', alpha=0.5)
			plt.plot(sub_part_x, 'r.-', alpha=0.5)
			plt.ylim(-6, 12)

		# last_part = (l_sel * (l_sel+1) - m_sel ** 2 / (np.sin(theta_range) ** 2)) * legendre
		last_part = ((np.sin(theta_range) ** 2) * l_sel * (l_sel+1) - m_sel ** 2) * legendre
		analytic_last_part = 18 * np.sin(theta_range) ** 2 - 12
		if plot_intermediate:
			plt.plot(analytic_last_part, 'b*-', alpha=0.5)
			plt.plot(last_part, 'r.-', alpha=0.5)

		test_eq = np.sin(theta_range) ** 2 * (sub_part_x + last_part)
		test_eq[np.isnan(test_eq)] = 0
		test_eq = test_eq[:-1]
		test_eq_analytic = analytic_legendre_theta_theta + analytic_last_part

		plt.figure()
		plt.plot(test_eq)
		plt.title(f'order{m_sel} degree {l_sel}')

"""

Now check the full spherical Bessel function equation thingy ...

"""

n = 100000
r_range = np.linspace(0, 1, n)
delta_r = np.diff(r_range)[0]
theta_range = np.linspace(0, 2 * np.pi, n)
delta_theta = np.diff(theta_range)[0]
x_range = np.cos(theta_range)
phi_range = np.linspace(0, np.pi, n)
delta_phi = np.diff(phi_range)[0]

order = 2
degree = 2
wave_number = 3.14
leg_pol = [scipy.special.lpmn(m=order, n=l, z=x)[0] for x in x_range]
leg_pol = np.array(leg_pol)

sph_bessel = scipy.special.jn(order+1/2, wave_number * r_range)
# R_factor = np.pi ** 0.5 * sph_bessel / (2 * r_range) ** 0.5
R_factor = sph_bessel / (wave_number * r_range) ** 0.5
R_factor_x = (1/delta_r) * np.gradient(R_factor, axis=0)
R_factor_xx = (1/delta_r) * np.gradient(R_factor_x, axis=0)

PHI_factor = np.exp(-1j * order * phi_range)
THETA_factor = leg_pol[:, order, degree]

z = (1 / R_factor) * (2 * r_range * R_factor_x + r_range ** 2 * R_factor_xx) + \
	wave_number ** 2 * r_range ** 2 + \
	(1/PHI_factor) * get_second_derivative(PHI_factor, delta_phi) * (1/np.sin(theta_range) ** 2) + \
	(1/THETA_factor) * (1/np.sin(theta_range)) * get_first_derivative(np.sin(theta_range) * get_first_derivative(THETA_factor, delta_theta), delta_theta)

plt.plot(z.real)
plt.ylim(-0.0001, 0.0001)

"""
Now we can compare how other packages do against our solution...
"""
scipy_sph_harm = scipy.special.sph_harm(order, degree, theta_range, phi_range)
ortho_sph_harm = orthopoly.spherical_harmonic.sph_har(t=phi_range, p=theta_range, n=degree, m=order)
np_sph_harm = PHI_factor * THETA_factor

plt.plot(scipy_sph_harm)
plt.plot(np_sph_harm, 'r')
plt.plot(PHI_factor, 'r.--')
plt.plot(THETA_factor, 'r--')
plt.plot(ortho_sph_harm, 'k')

# Check how he calculated the legendre polynomial
ortho_theta_factor = orthopoly.legendre.legen_theta(theta_range, n=degree, m=order)
plt.plot(THETA_factor)
plt.plot(ortho_theta_factor)
plt.plot(THETA_factor / ortho_theta_factor)

# Now get how he calculates the other angular part...
# Ye and here it is totally different
ortho_angular_part = ortho_sph_harm / ortho_theta_factor
plt.plot(ortho_angular_part)
plt.plot(PHI_factor)
plt.ylim(-3, 3)