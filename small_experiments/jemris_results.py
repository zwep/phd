

import scipy.io
ddata = '/home/bugger/Documents/data/jemris/tse_result.mat'
A = scipy.io.loadmat(ddata)
import matplotlib.pyplot as plt
plt.plot(A['t'], A['M'][:, 0])


# Did some simulations with X amount of repetitions.. Looking at the results now
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import re
ddata = '/home/bugger/Documents/data/jemris/Simulation_results_rep_1'
list_files = [os.path.join(ddata, x) for x in os.listdir(ddata) if x.endswith('h5')]
result = []
for i_file in sorted(list_files):
    flip_angle = re.findall("signal_output_([0-9]*)", i_file)[0]
    flip_angle = int(flip_angle)
    with h5py.File(i_file, 'r') as f:
        a = np.array(f['signal']['channels']['00'])
        b = np.array(f['signal']['times'])
        print('min max', b.min(), b.max())
    result.append({flip_angle: (a,b)})

result_sorted = sorted(result, key=lambda x: list(x.keys())[0])

for sel_position in range(40):
    fixed_position = []
    for i_result in result_sorted:
        for flip_angle, cont in i_result.items():
            a, b = cont
            print(flip_angle, a.shape, b.shape)
            # plt.plot(b, a[:,0])
            fixed_position.append([flip_angle, a[sel_position]])

    angle_distr_0 = np.array([[x[0], x[1][0]] for x in fixed_position])
    angle_distr_1 = np.array([[x[0], x[1][1]] for x in fixed_position])
    angle_distr_2 = np.array([[x[0], x[1][2]] for x in fixed_position])
    angle_variation = angle_distr_0[:, 0]
    signal_variation_0 = angle_distr_0[:, 1]
    signal_variation_1 = angle_distr_1[:, 1]
    signal_variation_2 = angle_distr_2[:, 1]
    plt.plot(angle_variation, np.sqrt(signal_variation_0 ** 2 + signal_variation_1 ** 2))
