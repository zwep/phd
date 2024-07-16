
# Folding...
import helper.array_transf as harray
import matplotlib.pyplot as plt
import helper.dummy_data as hdata
import skimage.data as skdata
import numpy as np
import helper.plot_class as hplotc
import multiprocessing
import multiprocessing as mp


# Unfolding...
def unfold(folded_image, reference_img, folding_factor, axis=-2):
    # Make complex here optional based on the input..
    unfolded_image = np.zeros(reference_img.shape[-2:], dtype=complex)

    if axis == -1:
        reference_img = np.swapaxes(reference_img, -2, -1)
        folded_image = np.swapaxes(folded_image, -2, -1)

    n_channel, n_x, n_y = reference_img.shape
    n_x_fold = int(n_x / folding_factor)

    for i_x in range(n_x_fold):
        for i_y in range(n_y):
            temp_signal = folded_image[:, i_x, i_y]
            temp_sens = reference_img[:, i_x::n_x_fold, i_y]
            temp_rho = np.matmul(np.linalg.pinv(temp_sens), temp_signal)
            unfolded_image[i_x::n_x_fold, i_y] = temp_rho

    if axis == -1:
        unfolded_image = np.swapaxes(unfolded_image, -2, -1)

    return unfolded_image


def unfold_mp(folded_image, reference_img, folding_factor, axis=-2):
    # Make complex here optional based on the input..
    unfolded_image = np.zeros(reference_img.shape[-2:], dtype=complex)

    if axis == -1:
        reference_img = np.swapaxes(reference_img, -2, -1)
        folded_image = np.swapaxes(folded_image, -2, -1)

    n_channel, n_x, n_y = reference_img.shape
    n_x_fold = int(n_x / folding_factor)

    res = []
    for i_x in range(n_x_fold):
        for i_y in range(n_y):
            temp_signal = folded_image[:, i_x, i_y]
            temp_sens = reference_img[:, i_x::n_x_fold, i_y]
            res.append((temp_sens, temp_signal))

    N = multiprocessing.cpu_count()
    with mp.Pool(processes=N) as p:
        results = p.map(calc_inv, res)

    counter = 0
    for i_x in range(n_x_fold):
        for i_y in range(n_y):
            unfolded_image[i_x::n_x_fold, i_y] = results[counter]
            counter += 1

    if axis == -1:
        unfolded_image = np.swapaxes(unfolded_image, -2, -1)

    return unfolded_image


def calc_inv(container):
    temp_sens, temp_signal = container
    temp_rho = np.matmul(np.linalg.pinv(temp_sens), temp_signal)
    return temp_rho


def fold(unfolded_image, reference_img, folding_factor, axis=-2):
    if axis == -1:
        unfolded_image = np.swapaxes(unfolded_image, -2, -1)
        reference_img = np.swapaxes(reference_img, -2, -1)

    n_channel, n_x, n_y = reference_img.shape
    n_x_fold = int(n_x / folding_factor)
    folded = np.empty((n_channel, n_x_fold, n_y))
    for i_x in range(n_x_fold):
        for i_y in range(n_y):
            temp_signal = unfolded_image[i_x::n_x_fold, i_y]
            temp_sens = reference_img[:, i_x::n_x_fold, i_y]
            temp_rho = np.matmul(temp_sens, temp_signal)
            folded[:, i_x, i_y] = temp_rho

    if axis == -1:
        folded = np.swapaxes(folded, -2, -1)

    return folded


if __name__ == "__main__":
    # Example of folding and unfolding of data..
    A_unfolded = skdata.camera()
    n_x, n_y = A_unfolded.shape
    A_unfolded = A_unfolded * hdata.get_elipse(n_x, n_y)
    A_reference = hdata.get_sentivitiy_maps(n_x, n_y, 4)
    n_channel = A_reference.shape[0]
    n_folding = min(2, n_channel)

    # Example using axes -1...
    A_folded = fold(A_unfolded, A_reference, n_folding, axis=-1)
    A_unfolded_rec = unfold(A_folded, A_reference, n_folding, axis=-1)
    hplotc.ListPlot([A_unfolded_rec, A_unfolded, A_folded, A_reference], augm='np.abs')

    # Example using axes -2...
    A_folded = fold(A_unfolded, A_reference, n_folding, axis=-2)
    import time
    t0 = time.time()
    A_unfolded_rec = unfold(A_folded, A_reference, n_folding, axis=-2)
    print(time.time() - t0)

    import time
    t0 = time.time()
    A_unfolded_rec = unfold_mp(A_folded, A_reference, n_folding, axis=-2)
    print(time.time() - t0)

    hplotc.ListPlot([A_unfolded_rec, A_unfolded, A_folded, A_reference], augm='np.abs')
