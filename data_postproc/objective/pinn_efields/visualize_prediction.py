import helper.plot_class as hplotc
import helper.misc as hmisc
import numpy as np
import matplotlib.pyplot as plt
import os
"""
Visualize remote predictions
"""

data = '/home/bugger/Documents/data/canbermeoved'
input_file = os.path.join(data, 'input.h5')
target_file = os.path.join(data, 'target.h5')
pred_file = os.path.join(data, 'pred.h5')
#
input_array = hmisc.load_array(input_file)
target_array = hmisc.load_array(target_file)[0]
pred_array = hmisc.load_array(pred_file)[0]
#
print(input_array.shape)
print(target_array.shape)
print(pred_array.shape)
#
hplotc.ListPlot([input_array])
hplotc.ListPlot([target_array])
hplotc.ListPlot([pred_array])