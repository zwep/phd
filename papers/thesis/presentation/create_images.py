import numpy as np
import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(20,5))
ax.yaxis.set_tick_params(labelleft=False)
ax.set_yticks([])

x_axis = np.arange(0.2, 14, 0.001)
notable_field_strengths = [0.2, 0.3, 0.4, 1, 1.5, 3, 4, 7, 9, 10.5, 11.75, 14]
gyromagnetic_ratio_dict = {'H-1': 42.58, 'Na-23': 11.26, 'P-31': 17.24}

for kernel, gyromagnetic_ratio in gyromagnetic_ratio_dict.items():
    if 'H-1' in kernel:
        larmor_freq_x = gyromagnetic_ratio * x_axis
        oscilation = np.sin(larmor_freq_x)
        ax.plot(x_axis, oscilation, label=kernel, color='k')
        ax.set_xscale('log')

ax.set_xticks(notable_field_strengths)
ax.set_xlim((min(x_axis), max(x_axis)+0.01))
ax.set_xticklabels(notable_field_strengths)
ax.set_xlabel('Field strength (T)')