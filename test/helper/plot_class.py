import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import helper.plot_class as hplotc

importlib.reload(hplotc)
A = np.random.rand(10, 10)
plot_obj = hplotc.PlotCollage(content_list=[A] * 16, plot_type='array', ddest='/home/bugger', n_display=6)
plot_obj.plot_collage()
#
# dtest = '/home/bugger/Pictures'
# file_list = [os.path.join(dtest, x) for x in os.listdir(dtest)]
# plot_obj = hplotc.PlotCollage(content_list=file_list[:10], ddest='/home/bugger', n_display=4, )
# plot_obj.plot_collage()