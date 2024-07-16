import helper.array_transf as harray
import numpy as np
import helper.plot_class as hplotc
import matplotlib.pyplot as plt


def get_t2_signal_simplified(flip_angle):
    return np.sin(flip_angle) ** 3


def get_t2_signal_parts(flip_angle, T1, TE, TR, beta=None, T2=None):
    # https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.20354
    # Beta is the angle of the refocussing pulse
    # Generic equation used in Wang 2005. Is the same as LASI when beta = 180 and T2 is not present.
    E1 = np.exp(-TR/T1)
    E11 = np.exp(TE/(2*T1))

    if beta is None:
        beta = flip_angle * 2
    numerator_term_1 = np.cos(beta) * E1
    numerator_term_2 = (1 - np.cos(beta)) * E1 * E11
    numerator = 1 - numerator_term_1 - numerator_term_2
    denominator_factor_1 = np.cos(flip_angle)
    denominator_factor_2 = np.cos(beta) * E1
    denominator = (1 - denominator_factor_1 * denominator_factor_2)
    signal_se = np.sin(flip_angle) * numerator / denominator
    if T2 is not None:
        signal_se = signal_se * np.exp(TE / T2)
    return signal_se, numerator_term_1, numerator_term_2, numerator, denominator_factor_1, denominator_factor_2, denominator


def get_t2_signal_general(flip_angle, T1, TR, TE, beta, N=1, T2=None):
    # https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.20354
    # Beta is the angle of the refocussing pulse
    # Generic equation used in Wang 2005. Is the same as LASI when beta = 180 and T2 is not present.
    E1 = np.exp(-TR/T1)
    numerator = 1 - (np.cos(beta)) ** N * E1 - M(N, beta, T1, TE, TR)
    denominator = (1 - np.cos(flip_angle) * (np.cos(beta)) ** N * E1)
    signal_se = np.sin(flip_angle) * numerator / denominator
    if T2 is not None:
        signal_se = signal_se * np.exp(TE / T2)
    return signal_se


def get_t2_signal_LASI(flip_angle, T1, TR, TE, T2=None):
    # LARGE ANGLE SPIN-ECHO IMAGING
    # N = 1, beta = 180
    # Now with added T2 factor..
    numerator = 1 - (2 * np.exp(-TR/T1) * np.exp(TE/(2*T1)) - np.exp(-TR/T1))
    denominator = 1 + np.cos(flip_angle) * np.exp(-TR/T1)
    signal_se = np.sin(flip_angle) * numerator / denominator
    if T2 is not None:
        signal_se = signal_se * np.exp(TE / T2)
    return signal_se


def M(N, beta, T1, TE, TR):
    result = 0
    for m in range(1, N+1):
        e_exponent = -(TR - (2 * m - 1) * TE/2)/T1
        temp = (1 - np.cos(beta)) * (np.cos(beta)) ** (N-m) * np.exp(e_exponent)
        result += temp
    return result


