import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile


dir_data = r"C:\Users\sebha\Documents\data\snoring\archive\Snoring Dataset\train"

sel_file = os.listdir(dir_data)[0]
sel_file_path = os.path.join(dir_data, sel_file)

rate, res = scipy.io.wavfile.read(sel_file_path)

plt.plot(np.abs(res[:, 0] + 1j * res[:, 1]))

# Okay.. I need to convert them to spectrograms...