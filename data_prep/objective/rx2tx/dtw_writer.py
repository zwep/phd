import time
import dtaidistance
import data_generator.Rx2Tx as gen_rx2tx
import numpy as np
import importlib
import multiprocessing as mp
import matplotlib.pyplot as plt


def multi_run_wrapper(args):
    temp = dtaidistance.dtw.distance(*args)
    return temp


# dir_data = '/home/bugger/Documents/data/7T/cardiac/b1shimsurv_all_channels'
dir_data = '/home/seb/data/b1shimsurv_all_channels'

# Test SURVEY2B1_all - line by line view
importlib.reload(gen_rx2tx)
dg_gen_rx2tx = gen_rx2tx.DataSetSurvey2B1_all(input_shape=(16, 512, 256), ddata=dir_data,
                                              input_is_output=False, number_of_examples=2,
                                              transform_type='complex', complex_type='cartesian', shuffle=False,
                                              relative_phase=False, masked=True,
                                              fourier_transform=False)

a, b = dg_gen_rx2tx.__getitem__(1)

N = mp.cpu_count()

fig_real_input = plt.figure()
plt.imshow(a[0])
plt.title('real input')
fig_real_input.savefig(dir_data + '/real_input.png')
np.save(dir_data + '/real_input.npy', a[0])

fig_real_target = plt.figure()
plt.imshow(b[0])
plt.title('real target')
fig_real_target.savefig(dir_data + '/real_target.png')
np.save(dir_data + '/real_target.npy', b[0])

fig_imag_input = plt.figure()
plt.imshow(a[1])
plt.title('imag input')
fig_imag_input.savefig(dir_data + '/imag_input.png')
np.save(dir_data + '/imag_input.npy', a[1])

fig_imag_output = plt.figure()
plt.imshow(b[1])
plt.title('imag target')
fig_imag_output.savefig(dir_data + '/imag_target.png')
np.save(dir_data + '/imag_target.npy', b[1])

Ny = 512
dtw_dist_real_real = np.zeros((Ny, Ny))
dtw_dist_imag_imag = np.zeros((Ny, Ny))
dtw_dist_real_imag = np.zeros((Ny, Ny))
dtw_dist_imag_real = np.zeros((Ny, Ny))
t0 = time.time()
for i in range(Ny):
    print(i, '  ------  ', time.time() - t0)
    with mp.Pool(processes=N) as p:
        results_real_real = p.map(multi_run_wrapper, [(a[0][i, :], b[0][j, :]) for j in range(Ny)])
        results_imag_imag = p.map(multi_run_wrapper, [(a[1][i, :], b[1][j, :]) for j in range(Ny)])
        results_real_imag = p.map(multi_run_wrapper, [(a[0][i, :], b[1][j, :]) for j in range(Ny)])
        results_imag_real = p.map(multi_run_wrapper, [(a[1][i, :], b[0][j, :]) for j in range(Ny)])

    dtw_dist_real_real[i] = results_real_real
    dtw_dist_imag_imag[i] = results_imag_imag
    dtw_dist_real_imag[i] = results_real_imag
    dtw_dist_imag_real[i] = results_imag_real

    np.save(dir_data + '/dtw_dist_real_real.npy', dtw_dist_real_real)
    np.save(dir_data + '/dtw_dist_imag_imag.npy', dtw_dist_imag_imag)
    np.save(dir_data + '/dtw_dist_real_imag.npy', dtw_dist_real_imag)
    np.save(dir_data + '/dtw_dist_imag_real.npy', dtw_dist_imag_real)

t_end = time.time()
print(t_end - t0)

np.save(dir_data + '/dtw_dist_real_real.npy', dtw_dist_real_real)
np.save(dir_data + '/dtw_dist_imag_imag.npy', dtw_dist_imag_imag)
np.save(dir_data + '/dtw_dist_real_imag.npy', dtw_dist_real_imag)
np.save(dir_data + '/dtw_dist_imag_real.npy', dtw_dist_imag_real)
print('Thing is saved')

fig_real_real = plt.figure()
plt.imshow(dtw_dist_real_real)
plt.title('real real')
fig_real_input.savefig(dir_data + '/real_real.png')
np.save(dir_data + '/real_real.npy', dtw_dist_real_real)

fig_imag_imag = plt.figure()
plt.imshow(dtw_dist_imag_imag)
plt.title('imag imag')
fig_imag_imag.savefig(dir_data + '/imag_imag.png')
np.save(dir_data + '/imag_imag.npy', dtw_dist_imag_imag)

fig_real_imag = plt.figure()
plt.imshow(dtw_dist_real_imag)
plt.title('real imag')
fig_real_imag.savefig(dir_data + '/real_imag.png')
np.save(dir_data + '/real_imag.npy', dtw_dist_real_imag)

fig_imag_real = plt.figure()
plt.imshow(dtw_dist_imag_real)
plt.title('imag real')
fig_imag_real.savefig(dir_data + '/imag_real.png')
np.save(dir_data + '/imag_real.npy', dtw_dist_imag_real)
