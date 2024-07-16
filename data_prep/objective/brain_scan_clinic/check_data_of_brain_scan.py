
import os
import numpy as np
import helper.plot_class as hplotc
import reconstruction.ReadListData as read_dl

"""
"""

ddata = '/media/bugger/WORK_USB/scan'

list_files = os.listdir(ddata)
dotlist_files = [os.path.join(ddata, x) for x in list_files if x.endswith('list')]
dotdata_files = [os.path.join(ddata, x) for x in list_files if x.endswith('data')]

for i_file in dotdata_files:
    print(i_file)
    print(round(os.path.getsize(i_file) / 1024 / 1024 / 1024, 2), ' Gb')


# The first one is able to be put into memory...
# The last one not...
i_file = dotlist_files[1]
import importlib
dl_obj = read_dl.DataListImage(input_file=i_file)
derp = dl_obj.get_image_data(sel_loc=0)
a, b = derp
hplotc.SlidingPlot(b.sum(axis=4))
# Maybe do some more with selecting a specific locatino....

