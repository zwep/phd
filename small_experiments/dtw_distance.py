
"""
Not sure what I did here... but it might be useful to save it

"""
import scipy.stats
import scipy.special
import scipy.spatial
import dtaidistance

plt.imshow(scipy.special.kl_div(a[0], b[0]))
plt.plot(scipy.special.kl_div(a[0][2, :], b[0][2, :]))
scipy.stats.entropy(a[0][0, :], b[0][0, :])

Ny = 512
dtw_dist = np.zeros((Ny, Ny))
for i in range(Ny):
    print(i)
    for j in range(Ny):
        dtw_dist[i, j] = dtaidistance.dtw.distance(a[0][i, :], b[0][j, :])

plt.plot(dtw_dist[0])

for n_line in range(0, a.shape[1], 100):
    plt.figure()
    plt.plot((a[0][n_line, :]), 'r', label='real', alpha=0.6)
    plt.plot((a[1][n_line, :]), 'r.-', label='imag', alpha=0.6)
    plt.plot((b[0][n_line, :]), 'b', label='real', alpha=0.6)
    plt.plot((b[1][n_line, :]), 'b.-', label='imag', alpha=0.6)
    plt.suptitle('masked')
    plt.legend()

for n_line in range(0, a.shape[2], 100):
    plt.figure()
    plt.plot((a[0][:, n_line]), 'r', label='real', alpha=0.6)
    plt.plot((a[1][:, n_line]), 'r.-', label='imag', alpha=0.6)
    plt.plot((b[0][:, n_line]), 'b', label='real', alpha=0.6)
    plt.plot((b[1][:, n_line]), 'b.-', label='imag', alpha=0.6)
    plt.suptitle('masked')
    plt.legend()

import torch.nn

A = torch.nn.Embedding(10, 200)
A.weight.shape


def get_gaussian_circle(mean, cov, N, val=0.1):
    # mean = [0,0]
    # N = 200
    # cov = [[N, 0],
    #        [0, N]]
    sample_gaus = np.random.multivariate_normal(mean, cov, size=N ** 2)

    x = sample_gaus[:, 0]
    y = sample_gaus[:, 1]
    delta_x = (np.max(x) - np.min(x)) / N
    y_N = int((np.max(y) - np.min(y)) // delta_x)

    A = np.zeros((N + 1, y_N + 1))
    for ix, iy in sample_gaus:
        pos_x = int((ix - np.min(x)) // delta_x)
        pos_y = int((iy - np.min(y)) // delta_x)
        A[pos_x, pos_y] += val

    # import matplotlib.pyplot as plt
    # plt.imshow(A)
    return A


def sub_matrix(A, B, offset=(0, 0)):
    # Put A into B...
    x_N, y_N = A.shape
    x_N_B, y_N_B = B.shape
    mov_x, mov_y = offset

    for i in range(x_N):
        for j in range(y_N):
            i_temp = i - x_N // 2 + mov_x - 1
            j_temp = j - y_N // 2 + mov_y - 1

            i_new = int(i_temp) if x_N_B > i_temp >= 0 else False
            j_new = int(j_temp) if y_N_B > j_temp >= 0 else False
            try:
                B[i_new, j_new] = A[i, j]
            except:
                pass
    return B


A = get_gaussian_circle(mean=[0, 1], cov=[[1, 5], [5, 1]], N=100)
B = np.zeros((200, 200))
B = sub_matrix(A, B, offset=(180, 180))

A = get_gaussian_circle(mean=[5, 1], cov=[[1, 5], [5, 1]], N=100)
C = np.zeros((200, 200))
C = sub_matrix(A, B, offset=(90, 90))
plt.imshow(C)
plt.imshow(B)

C = np.zeros((64, 64))
C[0:10, 0:10] = 1
B = np.copy(np.flipud(C))
plt.imshow(B)

conv_1 = torch.nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, groups=2)
conv_1_sample = torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=2, groups=2, stride=2)
conv_2 = torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, groups=2)
conv_2_sample = torch.nn.Conv2d(in_channels=2, out_channels=16, kernel_size=2, groups=2, stride=2)
