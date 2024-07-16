import numpy as np

import scipy.signal


def strideConv(arr, arr2, s):
    return scipy.signal.convolve2d(arr, arr2[::-1, ::-1], mode='valid')[::s, ::s]


# Trying to reverse the calculations...

A = np.array([[3, 2, 0],
              [3, 2, 1],
              [2, 3, 2]])

A_ravel = A.ravel()
A_result = np.array([[A_ravel[0], A_ravel[1], A_ravel[3], A_ravel[4]],
                     [A_ravel[1], A_ravel[2], A_ravel[4], A_ravel[5]],
                     [A_ravel[3], A_ravel[4], A_ravel[6], A_ravel[7]],
                     [A_ravel[4], A_ravel[5], A_ravel[7], A_ravel[8]]])

print('Original matrix', np.linalg.det(A))
print('Resulting matrix', np.linalg.det(A_result))
desired_result = np.array([4, 6, 6, 8])
resulting_kernel = np.linalg.solve(A_result, desired_result).reshape(2, 2).astype(int)
print('Kernel to use', resulting_kernel)
resulting_result = strideConv(A, resulting_kernel, 1).astype(int)
# Results can differ due to rounding errors
print('Check result ', resulting_result)

A_kernel = np.array([[-2, 2], [2, 0]])
B_kernel = np.array([[0, -3], [1, 3]])

avg_kernel = np.array([[1, 1], [1, 1]]) / 4

# First output
strideConv(A, A_kernel, 1)
strideConv(A, B_kernel, 1)

# Second output
strideConv(strideConv(A, A_kernel, 1), avg_kernel, 2)
strideConv(strideConv(A, B_kernel, 1), avg_kernel, 2)
