import os
import numpy as np
import skimage.data
import helper.plot_class as hplotc

"""

Super pixel augmentation

"""

A = skimage.data.astronaut()[..., 0]
A_backup = np.copy(A)
#

Nx, Ny = A.shape
center_point = np.array([Nx // 2, Ny // 2])
N_points = 100
x_coord = np.random.randint(0, Nx, N_points)
y_coord = np.random.randint(0, Ny, N_points)

max_dist = np.linalg.norm(center_point)

sel_index = 0
for sel_index in range(len(x_coord)):
    # Determine size of square...
    sel_coord = np.array([x_coord[sel_index], y_coord[sel_index]])
    dist_center_coord = np.linalg.norm(center_point - sel_coord)
    p = 0.14 * (dist_center_coord / max_dist)
    p = (p + 1) ** 3
    p = p - 1
    max_size = int(p * min(Nx, Ny))
    # print(p, sel_size)
    if max_size:
        sel_size = np.random.choice(range(0, max_size), 1)[0]
        min_x = max(0, x_coord[sel_index] - sel_size // 2)
        max_x = min(Nx, x_coord[sel_index] + sel_size // 2)
        min_y = max(0, y_coord[sel_index] - sel_size // 2)
        max_y = min(Ny, y_coord[sel_index] + sel_size // 2)
        A[min_x: max_x, min_y: max_y] = np.mean(A_backup[min_x: max_x, min_y: max_y])


fig_obj = hplotc.ListPlot(A)


# fig_obj.ax_list[0].vlines(min_y-1, min_x, max_x)
# fig_obj.ax_list[0].vlines(max_y+1, min_x, max_x)
# fig_obj.ax_list[0].hlines(min_x-1, min_y, max_y)
#fig_obj.ax_list[0].hlines(max_x+1, min_y, max_y)

import matplotlib.pyplot as plt
phi_range = np.arange(0, 4 * np.pi, 0.1)
alpha = 2
x_range = alpha * np.sqrt(phi_range) * np.cos(phi_range)
y_range = alpha * np.sqrt(phi_range) * np.sin(phi_range)
plt.scatter(x_range, y_range)
plt.scatter(-x_range, -y_range)

import numpy as np
from scipy.optimize import fsolve


# Define the equation to be solved
def equation_to_solve(phi_range_alpha, x_range):
    phi_range, alpha = phi_range_alpha
    y = x_range / alpha
    return phi_range - y**2 * (1 / np.cos(phi_range))**2


def eval_equation(phi_range, alpha):
    return alpha * np.sqrt(phi_range) * np.cos(phi_range)

# Define parameters
alpha = 1.0  # Replace 1.0 with the actual value of alpha
x_range = 2.0  # Replace 2.0 with the actual value of x_range

# Initial guess for phi_range
initial_guess = (0.5, 1)  # Replace 0.5 with a reasonable initial guess

# Solve the equation numerically
phi_range_solution = fsolve(equation_to_solve, initial_guess, args=(x_range))

print("phi_range:", phi_range_solution[0])

eval_equation(phi_range_solution[0], 1)
equation_to_solve(-2, 2, 1)

eval_equation(equation_to_solve(-2, 2, 1), 1)