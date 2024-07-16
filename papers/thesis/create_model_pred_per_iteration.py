import os
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
from objective_configuration.reconstruction import DDATA
import objective_helper.reconstruction as hrecon
import numpy as np
import re

"""

To visualize how something learns...

We trained a Unet using X amount of data and stored the model weights every 5 steps.

In evaluate_model.sh we set

        data_path="${DDATA}/${anatomy}/test/input"

to

        data_path="${DDATA}/${anatomy}/test/input_25"

So that inference time is fast.
Then we execute

papers/reconstruction/DIRECT/evaluate_model.sh -m unet_SCRATCH_ACQ_test -p 100 -o
python data_postproc/objective/reconstruction/h5_to_png.py -path /home/sharreve/local_scratch/paper/reconstruction/results/unet_SCRATCH_ACQ_test/100p/train_mixed/mixed/5x -key reconstruction

and each time change the model_number_deprecated in the model folder of unet_SCRATCH_ACQ

This resulted in a couple of model predictions, stored in

/home/sharreve/local_scratch/paper/reconstruction/results/unet_SCRATCH_ACQ_test/100p/train_mixed/mixed/

Now we are going to visualize those predictions nicely

Also the full trained model is at

/home/sharreve/local_scratch/paper/reconstruction/results/unet_SCRATCH_ACQ/100p/train_mixed/mixed/5x
"""


def get_input_target(i_file):
    input_array = hmisc.load_array(os.path.join(ddata_input, i_file), data_key='kspace', sel_slice='mid')
    n_points = input_array.shape[0] * 2
    us_radial_traj = hrecon.undersample_trajectory(img_size=input_array.shape[:2], n_points=n_points,
                                                   p_undersample=100 // 5)
    temp_cpx = input_array[..., ::2] + 1j * input_array[..., 1::2]
    input_cpx = np.fft.ifftn(np.fft.fftshift(temp_cpx), axes=(-3, -2))
    input_cpx = np.moveaxis(input_cpx, -1, 0)
    input_array = hrecon.undersample_img(input_cpx, traj=us_radial_traj)
    input_sos = np.sqrt(np.sum(np.abs(input_array) ** 2, axis=0))
#
    target_array = hmisc.load_array(os.path.join(ddata_target, i_file), data_key='kspace', sel_slice='mid')
    target_sos = hrecon.convert_to_sos(target_array)
    return input_sos, target_sos


ddata = ddata_input = ddata_target = os.path.join(DDATA, 'mixed', 'test', 'input')
acc = 5
acc_str = '5x'
diter = '/home/sharreve/local_scratch/paper/reconstruction/results/unet_SCRATCH_ACQ_test/100p/train_mixed/mixed/'
list_iterdir = [os.path.join(diter, x) for x in os.listdir(diter) if x.startswith('5x')]

sel_dir = list_iterdir[0]
file_list = [x for x in os.listdir(sel_dir) if x.endswith('h5')]
hplotc.close_all()
temp_input, temp_target = zip(*[get_input_target(x) for x in file_list])
temp_input = np.array([x[::-1, ::-1] for x in temp_input])
temp_target = np.array([x[::-1, ::-1] for x in temp_target])

all_plot = [temp_input]
iter_num_list = []
for i_dir in sorted(list_iterdir, key=lambda x: int(re.findall('5x_([0-9]+)', x)[0])):
    temp_plot = []
    iter_num = re.findall('5x_([0-9]+)', i_dir)[0]
    iter_num_list.append(iter_num)
    ddest = os.path.join(diter, f'plot_{iter_num}')
    for i_file in file_list:
        dd = os.path.join(i_dir, i_file)
        A = hmisc.load_array(dd, data_key='reconstruction', sel_slice='mid')
        print(A.shape)
        temp_plot.append(A)
    #
    #[x.shape for x in temp_plot]
    temp_plot = [x[::-1, ::-1] for x in temp_plot]
    all_plot.append(temp_plot)
    plot_obj = hplotc.ListPlot(temp_plot, ax_off=True, col_row=(3,1),
                               proper_scaling=False, proper_scaling_patch_shape=64, figsize=(8,4), hspace=0.01)
    plot_obj.savefig(ddest, home=False)

all_plot.append(temp_target)
ddest = os.path.join(diter, f'plot_all')


# How to get the input...
temp = np.array(all_plot)
plot_obj = hplotc.ListPlot([[y for y in x] for x in temp], ax_off=True, wspace=0.0, aspect='auto',
                               proper_scaling=True, proper_scaling_patch_shape=128, figsize=(15,15), hspace=0.00)
plot_obj.savefig(ddest, home=False)


""" Different order """

text_box = ['Input'] + ['Iteration ' + x for x in iter_num_list] + ['Target']
plot_obj = hplotc.PlotCollage([x for x in temp[:, 1]], ddest=diter, plot_type='array', n_display=len(temp), proper_scaling=True,
                                proper_scaling_patch_shape=128,
                              subtitle_list=text_box)
plot_obj.plot_collage()

text_box = ['Input'] + ['Iteration ' + x for x in iter_num_list] + ['Target']
plot_obj = hplotc.PlotCollage([x for x in temp[:, 1]], ddest=diter, plot_type='array', n_display=len(temp), proper_scaling=True,
                                proper_scaling_patch_shape=128, sub_col_row=(4,2),
                              subtitle_list=text_box)
plot_obj.plot_collage()
