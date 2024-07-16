from scipy.spatial.distance import directed_hausdorff
import numpy as np
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import matplotlib.pyplot as plt

"""
Derpederp

Do we calculate the hausdorf distance correctly?

WE DID NOT. OMG So many mistakes
"""

# A = np.array([[1,0,0], [0,0,0], [0,0,0]])
# B = np.array([[0,0,0], [0,0,0], [0,0,1]])
for _ in range(100):
    A = np.random.randint(0, 2, size=(50, 50))
    B = np.random.randint(0, 2, size=(50, 50))

    hd_based_on_binary_ab = directed_hausdorff(A, B)[0]
    hd_based_on_binary_ba = directed_hausdorff(A, B)[0]
    hd_based_on_binary = max(hd_based_on_binary_ab, hd_based_on_binary_ba)

    A_index = np.argwhere(A)
    B_index = np.argwhere(B)
    hd_based_on_index_ab = directed_hausdorff(A_index, B_index)[0]
    hd_based_on_index_ba = directed_hausdorff(A_index, B_index)[0]
    hd_based_on_index = max(hd_based_on_index_ab, hd_based_on_index_ba)

    if hd_based_on_index != hd_based_on_binary:
        print(hd_based_on_index, hd_based_on_binary)


"""
Redo their example
"""

u = np.array([(1.0, 0.0),
              (0.0, 1.0),
              (-1.0, 0.0),
              (0.0, -1.0)])
v = np.array([(2.0, 0.0),
              (0.0, 2.0),
              (-2.0, 0.0),
              (0.0, -4.0)])

# Convert the coordinates to a matrix..
#
C = np.zeros((7, 7))
for i_point in u:
    i_point = i_point.astype(int)
    C[i_point[0]+4, i_point[1]+4] = 1

D = np.zeros((7, 7))
for i_point in v:
    i_point = i_point.astype(int)
    D[i_point[0] + 4, i_point[1] + 4] = 1


hausdorf_with_index = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
hausdorf_with_array = max(directed_hausdorff(C, D)[0], directed_hausdorff(D, C)[0])
print(hausdorf_with_index, hausdorf_with_array)