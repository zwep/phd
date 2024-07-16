import numpy as np
from skimage.util import img_as_ubyte, img_as_int, img_as_uint
import helper.array_transf as harray
from PIL import Image
import helper.misc as hmisc
import os
import imageio#.v2 as imageio

"""
We are going to get a lot of predictions... and we need to visualize that in a GIF I guess
"""

ddata = '/home/sharreve/local_scratch/paper/reconstruction/free_breathing/xpdnet_PRETR_SYNTH_ACQ/100p/train_mixed/undersampled/'
output_gif_dir = '/home/sharreve/local_scratch/paper/reconstruction/free_breathing_png'
os.makedirs(output_gif_dir, exist_ok=True)

data_key = 'reconstruction'
file_list = os.listdir(ddata)
file_list = [x for x in file_list if not x.startswith('collage') and x.endswith('h5')]
import helper.plot_class as hplotc

sorted_names = sorted(file_list, key=lambda x: (int(hmisc.get_base_name(x).split('_')[2]), int(hmisc.get_base_name(x).split('_')[3])))

sorted(file_list)
for ii, sel_file in enumerate(file_list):
    temp_array = hmisc.load_array(os.path.join(ddata, sel_file), data_key=data_key, sel_slice='mid')
    temp_array = img_as_uint(harray.scale_minmax(temp_array))
    image = Image.fromarray(temp_array)
    image.save(os.path.join(output_gif_dir, f'frame_{ii}.png'))
#!@
# fig_obj = hplotc.ListPlot(temp_array)
# fig_obj.savefig('check')

image_files = [os.path.join(output_gif_dir, f'frame_{i}.png') for i in range(len(file_list))]
imageio.mimsave(os.path.join(output_gif_dir, 'output.gif'), [imageio.imread(file) for file in image_files], duration=0.1)
