import helper.plot_fun as hplotf
import numpy as np
import importlib
import matplotlib.pyplot as plt
import helper.plot_class as hplotc

importlib.reload(hplotf)
A = np.random.rand(10, 10)
fig, ax = plt.subplots()
imshow_ax = ax.imshow(A, aspect='auto')
hplotf.add_text_box(fig, 0, 'test', linewidth=1)