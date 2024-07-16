import helper.plot_class as hplotc
import numpy as np
import helper.misc as hmisc
import helper.array_transf as harray
import os

from objective_configuration.thesis import DPLOT

dd = os.path.join(DPLOT, 'Comparisson Field strength/Cardiac')
os.makedirs(dd, exist_ok=True)

dd1 = '/home/bugger/Documents/paper/thesis/Figures/Comparisson Field strength/Cardiac/'
file_1p5T = os.path.join(dd1, 'example_1p5T.dcm')
file_3T = os.path.join(dd1, 'example_3T.dcm')
file_7T = os.path.join(dd1, 'example_7T.dcm')

array_1p5T = hmisc.load_array(file_1p5T)[None]
array_3T = hmisc.load_array(file_3T)[15:16, ::-1]
array_7T = hmisc.load_array(file_7T)[15:16, ::-1]

plot_array = [array_1p5T, array_3T, array_7T]
plot_obj = hplotc.ListPlot(plot_array, proper_scaling=True, col_row=(3, 1), ax_off=True, wspace=0)
plot_obj.savefig(os.path.join(dd, 'test'), home=False)

