
import matplotlib.pyplot as plt
import numpy as np
import os

gyromagnetic_ratio = {"1H": 42.58,
                      "23Na": 11.262,
                      "13C": 10.7084,
                      "7Li": 16.546,
                      "31P": 17.235}

field_strength = np.arange(0, 4, 0.1)
fig, ax = plt.subplots()
for k, v in gyromagnetic_ratio.items():
    line_obj = ax.plot(field_strength, field_strength * v, label=k)
    ax.scatter(1.5, 1.5 * v, color=line_obj[0]._color)
    ax.scatter(3, 3 * v, color=line_obj[0]._color)
    ax.hlines(1.5 * v, xmin=1.5, xmax=3, color=line_obj[0]._color)
    #ax.hlines(3 * v, xmin=1.5, xmax=3, color=line_obj[0]._color)
plt.legend()