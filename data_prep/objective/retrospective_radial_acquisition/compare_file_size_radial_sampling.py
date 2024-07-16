"""
Radial sampling..
"""

import os

ddata = '/media/bugger/MyBook/data/7T_scan/cardiac'

size_dict = {}
for d, _, f in os.walk(ddata):
    if len(f):
        filter_f = [x for x in f if x.endswith('raw') and 'transradialfast' in x]
        if len(filter_f):
            for ix in filter_f:
                file_path = os.path.join(d, ix)
                # file_size = os.path.getsize(file_path)
                file_size = round(os.path.getsize(file_path) / 1024 / 1024, 2)
                size_dict[file_path] = file_size

import helper.misc as hmisc
hmisc.print_dict(size_dict)
zz = [(k, v) for k, v in size_dict.items()]
for filename, size in sorted(zz, key=lambda x: x[1]):
    tempname = os.path.basename(filename)
    print(tempname, (50 - len(tempname)) * ' ', size)



# Why is this one so low...?
# Because it only has 3 heartphases.....
import reconstruction.ReadCpx as read_cpx
import helper.plot_class as hplotc
ddata = '/media/bugger/MyBook/data/7T_scan/cardiac/2020_10_14/V9_13296/v9_14102020_1751406_17_3_transradialfastV4.cpx'
img = read_cpx.ReadCpx(ddata).get_cpx_img()
hplotc.SlidingPlot(img.sum(axis=0))