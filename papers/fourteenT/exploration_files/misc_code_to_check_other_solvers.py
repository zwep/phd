
import sys

sys.path.append('/')
import scipy.sparse.linalg
import os
import numpy as np
from objective_helper.fourteenT import ReadMatData, OptimizeData

"""
Here we do a lot of stuff...
"""

ddata = '/home/bugger/Documents/data/14T'
ddest = '/home/bugger/Documents/paper/14T/plots_no_normalization'

file_list = os.listdir(ddata)
mat_files = [x for x in file_list if x.endswith('mat')]

for sel_mat_file in mat_files:
    print(sel_mat_file)
    pass

mat_reader = ReadMatData(ddata=ddata, mat_file=sel_mat_file)
data_obj = OptimizeData(ddest=ddest, mat_reader=mat_reader, normalization=False)

data_obj.lambda_range = np.logspace(-2, 0, num=50)
cond_list, sel_slice = data_obj.get_min_cond_slice(mask_array=data_obj.sigma_mask)

# # Checking other problem solvers..
sel_lambda = 0
sel_slice = data_obj.brain_mask.shape[0] // 2
sel_mask = data_obj.brain_mask[sel_slice]

system_matrix = data_obj.get_masked_flat_b1p(sel_slice=sel_slice, mask_array=sel_mask)
A_static = system_matrix.T.conjugate() @ system_matrix
result_dict = data_obj.solve_ridge_regression(lambda_value=sel_lambda, sel_slice=sel_slice, mask_array=sel_mask)
print(result_dict['b1p_nrmse'])
print(result_dict['cond'])
# Now try it with GC without preconditioning
system_matrix = data_obj.get_masked_flat_b1p(sel_slice=sel_slice, mask_array=sel_mask)
A_static = system_matrix.T.conjugate() @ system_matrix
cond = np.linalg.cond(A_static)
b = system_matrix.T.conjugate() @ np.ones(system_matrix.shape[0])
solution, _ = scipy.sparse.linalg.cg(A_static, b)
result_dict = data_obj.get_result_container(solution, sel_slice=sel_slice, mask_array=sel_mask)
print(result_dict['b1p_nrmse'])
print(cond)
# Now try it WITH preconditioning
eig_value, eig_vector = np.linalg.eig(A_static)
D = np.diag(eig_value)
# Check if reducing this will make the condition number better...
for sel_k in range(0, 8):
    A_approx = eig_vector[:, :sel_k] @ D[:sel_k, :sel_k] @ eig_vector.T.conjugate()[:sel_k, :]
    # A_approx = eig_vector[:, sel_k:] @ D[sel_k:, sel_k:] @ eig_vector.T.conjugate()[sel_k:, :]
    print(np.linalg.cond(A_approx))

M = np.diag(np.diag(A_static))
M = np.diag(np.diag(A)) + np.diag(np.diag(A, k=1), k=1) + np.diag(np.diag(A, k=-1), k=-1)
solution, _ = scipy.sparse.linalg.cg(A_static, b, M=np.linalg.inv(M))
result_dict = data_obj.get_result_container(solution, sel_slice=sel_slice, mask_array=sel_mask)
print(result_dict['b1p_nrmse'])
print(cond)
# And what if I do this preconditioning myself and use the closed form solution..?
# We use as preconditioner the diagonal..
M = np.diag(np.diag(A_static))
L = np.linalg.cholesky(M)
L_inv = np.linalg.inv(L)
B = (L_inv @ A_static @ L_inv.T)
b = system_matrix.T.conjugate() @ np.ones(system_matrix.shape[0])
target_b = L_inv @ b
y_shim = (np.linalg.inv(B.T.conjugate() @ B) @ B.conjugate().T @ (target_b))
x_shim = L_inv.T @ y_shim
result_dict = data_obj.get_result_container(x_shim, sel_slice=sel_slice, mask_array=sel_mask)
print(result_dict['b1p_nrmse'])
print(cond)

# Try ith with LDU factorization....
M = np.diag(np.diag(A_static))
(P, L, U) = scipy.linalg.lu(M)
D = np.diag(np.diag(U))
U /= np.diag(U)[:, None]
np.linalg.norm(P @ L @ D @ U - M)


# Try it with the SVD..
left_x, eig_x, right_x = np.linalg.svd(system_matrix, full_matrices=False)
A_approx = left_x @ np.diag(eig_x) @ right_x
A_pinv = right_x @ np.linalg.inv(np.diag(eig_x)) @ left_x.T
x_shim = A_pinv @ np.ones(A_pinv.shape[1])
result_dict = data_obj.get_result_container(x_shim, sel_slice=sel_slice, mask_array=sel_mask)
print(result_dict['b1p_nrmse'])

# Try ith with the SVD on
left_x, eig_x, right_x = np.linalg.svd(A_static, full_matrices=False)
M = left_x[:, :1] @ np.diag(eig_x[:1]) @ right_x[:1, :]
L = np.linalg.cholesky(M)
L_inv = np.linalg.inv(L)
B = (L_inv @ A_static @ L_inv.T)
b = system_matrix.T.conjugate() @ np.ones(system_matrix.shape[0])
target_b = L_inv @ b
y_shim = (np.linalg.inv(B.T.conjugate() @ B) @ B.conjugate().T @ (target_b))
x_shim = L_inv.T @ y_shim
result_dict = data_obj.get_result_container(x_shim, sel_slice=sel_slice, mask_array=sel_mask)
print(result_dict['b1p_nrmse'])
# #