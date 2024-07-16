import helper.misc as hmisc
import helper.array_transf as harray
import helper.plot_class as hplotc
import os
import numpy as np

from objective_configuration.reconstruction import DDATA, ANATOMY_LIST

"""
What is the effect of RSS on the data that we give..?

Well the effect is visible

But this operation cancels any complex output of course. We eradicate all the phase information.
Is that what we want...?

"""

sel_path = os.path.join(DDATA, ANATOMY_LIST[0], 'test', 'input')
file_list = os.listdir(sel_path)
sel_file = os.path.join(sel_path, file_list[0])

A = hmisc.load_array(sel_file, data_key='kspace')
print("Selected file\t", sel_file)
print("Shape \t", A.shape)
data_kspace = A[:, :, :, ::2] + 1j * A[:, :, :, ::2]
data = np.fft.ifftn(data_kspace, axes=(1, 2))
RSS = np.sqrt((data ** 2).sum(-1).sum(0))
ARSS = np.sqrt((np.abs(data) ** 2).sum(-1).sum(0))
plot_obj = hplotc.ListPlot([RSS, ARSS])
plot_obj.figure.savefig('/local_scratch/sharreve/test.png')