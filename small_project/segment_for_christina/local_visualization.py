import os
import helper.misc as hmisc
import helper.plot_class as hplotc

"""
We have processed everything remotely.. but I have a nice local 4D image visualizer

So we copied the data and can visualize it here..
"""

dsource_img = '/home/bugger/Documents/data/segment_4d/image'
dsource_pred = '/home/bugger/Documents/data/segment_4d/pred'

file_list = os.listdir(dsource_img)
for ii, i_file in enumerate(file_list):
    print(f'Possible file {ii} - {i_file}')

sel_file = file_list[0]

source_img_array = hmisc.load_array(os.path.join(dsource_img, sel_file)).T
source_pred_array = hmisc.load_array(os.path.join(dsource_pred, sel_file)).T

hplotc.SlidingPlot(source_img_array)
hplotc.SlidingPlot(source_pred_array)