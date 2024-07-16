import numpy as np
import matplotlib.pyplot as plt


def perturbed_trajectory(amplitude, frequency, alpha):
    """
    Used to generate a perturbed trajectory...

    :param amplitude: Amplitude of the deviation
    :param frequency: Frequency of the deviation
    :param alpha: Term for scaling the location of the pertubation
    :return:
    """
    lambda_alpha = np.sqrt(np.log(1 / alpha))
    lambda_amplitude = amplitude
    lambda_frequency = frequency
    modulate_amplitude = lambda_amplitude * np.exp(-(lambda_alpha * x_range) ** 2)
    frequency_amplitude = np.cos(lambda_frequency * np.pi * x_range)
    perturbed_trajectory = modulate_amplitude * frequency_amplitude
    return perturbed_trajectory

# Generic
N = 500
x_range = np.linspace(-1, 1, N)

# Get an amplitude modulation function
lambda_alpha = np.sqrt(np.log(1/0.001))
lambda_amplitude = 0.1
modulate_amplitude = lambda_amplitude * np.exp(-(lambda_alpha * x_range) ** 2)
plt.plot(x_range, modulate_amplitude)

# Get a frequency modulated function
lambda_frequency = 4
frequency_amplitude = np.cos(lambda_frequency * np.pi * x_range)
plt.plot(x_range, frequency_amplitude)

# Combine them
fig, ax = plt.subplots()
for i_amp in [0.05]:
    for i_freq in [4, 64]:
        for i_alpha in [1e-4, 1e-50]:
            traj_1 = perturbed_trajectory(i_amp, i_freq, i_alpha)
            ax.plot(traj_1)
#            ax.set_ylim(-1, 1)


print(__name__, __file__)