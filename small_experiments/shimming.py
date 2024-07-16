

import helper.array_transf as harray
import matplotlib.pyplot as plt
import numpy as np
import helper.plot_class as hplotc
import tooling.shimming.b1shimming_single as mb1


def improved_scale_signal_model(x, mask, flip_angle=np.pi / 2):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3310288/
    # Musculoskeletal MRI at 3.0T and 7.0T: A Comparison of Relaxation Times and Image Contrast
    # TR_se = 5000
    TR_se = 2500
    # TE_se = 50
    TE_se = 90
    T1_fat = 583
    T2_fat = 46
    T1_muscle = 1552
    T2_muscle = 23
    T1 = (T1_fat + T1_muscle) / 2
    T2 = (T2_fat + T2_muscle) / 2

    # Use a (binary) mask to determine the average signal
    x_sub = x * mask
    # Mean over the masked area...
    x_mean = np.abs(x_sub.sum()) / np.sum(mask)

    # Taking the absolute values to make sure that values are between 0..1
    # B1 plus interference by complex sum. Then using abs value to scale
    # Add some randomness to the flipangle....
    flip_angle = np.random.uniform(flip_angle - np.pi / 18, flip_angle + np.pi / 18)
    flip_angle_map = np.abs(x) / x_mean * flip_angle
    print('Mean flip angle map center ', flip_angle_map[mask==1].mean(), flip_angle_map.min(), flip_angle_map.max())
    general_signal_se = get_t2_signal_general(flip_angle=flip_angle_map,
                                                   T1=T1, TE=TE_se, TR=TR_se, T2=T2, N=1,
                                                   beta=flip_angle * 2)
    # Ik denk niet dat ik hier de angle van (x) ook nog eens nodig heb....
    # x_scaled_cpx = general_signal_se * np.exp(1j * np.angle(x))
    return general_signal_se, flip_angle_map


ddata = '/home/bugger/Documents/paper/inhomogeneity removal/data_creation/example_b1_plus_registrated.npy'
A = np.load(ddata)
b1_mask = np.zeros(A.shape[-2:])
n_y, n_x = A.shape[-2:]
y_center, x_center = (n_y // 2, n_x // 2)
delta_x = int(0.05 * n_y)
b1_mask[y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x] = 1

for _ in range(10):
    shim_proc = mb1.ShimmingProcedure(A, b1_mask)
    opt_shim, opt_value = shim_proc.find_optimum()
    new_array = harray.apply_shim(A, cpx_shim=opt_shim)
    orig_array = A.sum(axis=0)

    signal_array, fa_map = improved_scale_signal_model(new_array, b1_mask)

    hplotc.ListPlot(np.rad2deg(fa_map), cbar=True)
    plt.plot(harray.scale_minmax(np.rad2deg(fa_map)[256]))
    plt.plot(harray.scale_minmax(np.rad2deg(signal_array)[256]))

    signal_array_orig = improved_scale_signal_model(A.sum(axis=0), b1_mask)


    hplotc.ListPlot([A.sum(axis=0), new_array, signal_array_orig, signal_array], augm='np.abs', start_square_level=2)

