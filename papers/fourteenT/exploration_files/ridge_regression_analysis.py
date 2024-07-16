import sys
import scipy.sparse.linalg
import scipy.linalg

sys.path.append('/')
import os
import numpy as np
import matplotlib.pyplot as plt
from objective_helper.fourteenT import ReadMatData
import re


"""
The solutions we obtain... are unstable.. How unstalbe exactly?
And can a minimization procedure help us in this?

This script contains an analysis....
"""

ddata = '/home/bugger/Documents/data/14T'
ddest = '/home/bugger/Documents/paper/14T/plots'

file_list = os.listdir(ddata)
mat_files = [x for x in file_list if x.endswith('mat')]

sel_mat_file = mat_files[-1]
file_name = sel_mat_file.split('_')[0]
n_ports = int(re.findall('([0-9]+) Channel', sel_mat_file)[0])
mat_reader = ReadMatData(ddata=ddata, mat_file=sel_mat_file)
mat_container, mask_container = mat_reader.read_mat_object()

brain_mask = mask_container['target_mask'] - mask_container['substrate_mask']
n_slice = brain_mask.shape[0]
sel_slice = n_slice // 2
brain_mask = brain_mask[sel_slice]

# Mask creation
masked_b1 = mat_container['b1p'][:, sel_slice, brain_mask == 1].reshape((n_ports, -1)).T
masked_b1 = masked_b1 / np.linalg.norm(masked_b1)
# Calculate the shim settings for a range of values.. going from 0..1 in 20 steps on a log scale
lambda_range = np.logspace(-2, 0, 50, base=10)
result = []
N_mask = np.sum(brain_mask == 1)
# We have a closed form solution. A_static is a constant in this and doesnt depend on labda
A_static = masked_b1.T.conjugate() @ masked_b1
# # Testing for proper solution when condition number of matrix is bad.
sol_dict = {f'solution_{ii}': [] for ii in range(5)}
#
for lambda_value in lambda_range:
    A = A_static + lambda_value * np.eye(masked_b1.shape[1])
    cond_A = np.linalg.cond(A)
    print(np.round(lambda_value, 2), np.round(cond_A, 2), np.round(np.abs(np.linalg.det(A)),2))
    # What is this..
    tol = 1e-5
    solution_1, _ = scipy.sparse.linalg.gmres(A, np.ones(masked_b1.shape[1]))
    # Use a simple preconditioning matrix..?
    solution_2, _ = scipy.sparse.linalg.gmres(np.diag(np.diag(A)) @ A, np.diag(np.diag(A)) @ np.ones(masked_b1.shape[1]))
    solution_3, _ = scipy.sparse.linalg.cg(A, np.ones(masked_b1.shape[1]), M=np.diag(np.diag(A)))
    # Since we are working with complex numbers.. I need the complex conjugate
    solution_4 = (np.linalg.pinv(A) @ masked_b1.conjugate().T @ (np.ones(masked_b1.shape[0])))
    solution_5 = (np.linalg.inv(A) @ masked_b1.conjugate().T @ (np.ones(masked_b1.shape[0])))
    solution_list = [solution_1, solution_2, solution_3, solution_4, solution_5]
    for ii, i_sol in enumerate(solution_list):
        res = np.linalg.norm(masked_b1 @ i_sol - 1, ord=2)
        norm_sol = np.linalg.norm(i_sol, ord=2)
        sol_dict[f'solution_{ii}'].append((res, norm_sol))

# Plot the various solutions for lambda...
n_sol = len(sol_dict)
fig, ax = plt.subplots(n_sol)
style_list = [':', '-', '--', '-.', '--']
marker_list = ['8', 'v', 'o', '<', '>']
for i in range(len(sol_dict)):
    residual_list, norm_list = zip(*sol_dict[f'solution_{i}'])
    ax[i].plot(residual_list, norm_list, linestyle=style_list[i], marker=marker_list[i])

# Effects pre-conditioning..
A_static = masked_b1.T @ masked_b1
preconditioner = scipy.linalg.cholesky(A_static)
print(np.linalg.cond(A_static))
print(np.linalg.cond(np.diag(np.diag(A_static)) @ A_static))
print(np.linalg.cond(np.diag(np.diag(A_static)) @ A_static))
