"""

Get some radial data.. store it as a GIF
"""

import imageio
import numpy as np
import os
import helper.plot_class as hplotc
import helper.plot_fun as hplotf
import reconstruction.ReadCpx as read_cpx


derp = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_13/V9_17069/v9_13022021_1346189_7_3_transradialfast_high_timeV4.cpx'
target = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_13/V9_17069/fast_radial_gif.gif'

cpx_obj = read_cpx.ReadCpx(derp)
cpx_img = cpx_obj.get_cpx_img()
A = np.squeeze(np.rot90(cpx_img[-8:].sum(axis=0), k=2, axes=(-2, -1)))

imageio.mimsave(target, np.abs(A))

derp_notrig = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_13/V9_17069/v9_13022021_1347235_8_3_transradial_no_trigV4.cpx'
target_notrig = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_13/V9_17069/fast_radial_gif_notrig.gif'
cpx_obj = read_cpx.ReadCpx(derp_notrig)
cpx_obj.get_cpx_header()

cpx_img = cpx_obj.get_cpx_img()
A = np.squeeze(np.rot90(cpx_img[-8:].sum(axis=0), k=2, axes=(-2, -1)))
hplotc.SlidingPlot(A)
imageio.mimsave(target_notrig, np.abs(A), fps=5)

derp_cart = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_03/V9_16834/DICOM_6_1_Cine1SliceR2_tra/DICOM/IM_0002'
target_cart = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_02_03/V9_16834/cart.gif'
import pydicom
import helper.array_transf as harray
A = pydicom.read_file(derp_cart).pixel_array
A = harray.scale_minmax(A, axis=(-2, -1))
imageio.mimsave(target_cart, np.abs(A))


# Potentia colab with Yasmina/Sina
import os
import nibabel as nib
import numpy as np
import helper.plot_class as hplotc
path_2d = '/home/bugger/Documents/data/data_for_yasmina/7TCardiac/Segmented/2D'
path_3d = '/home/bugger/Documents/data/data_for_yasmina/7TCardiac/Segmented/3D'
path_input = '/home/bugger/Documents/data/data_for_yasmina/7TCardiac/Images'

sel_path = path_2d
file_list = [os.path.join(sel_path, x) for x in os.listdir(sel_path)]
file_array = [nib.load(x) for x in file_list]

single_2d = np.array(file_array[0].get_fdata())
hplotc.ListPlot(single_2d)
mult_2d = np.array(file_array[1].get_fdata())
hplotc.SlidingPlot(mult_2d)

# Object below is loaded from `model_based_on_measured` script
import torch
import helper.array_transf as harray
single_2d_scaled = harray.scale_minpercentile(single_2d, 98)
single_2d_scaled = harray.scale_minmax(single_2d_scaled)
a_tens = torch.from_numpy(np.tile(single_2d_scaled, (16, 1, 1))).float()
res = modelrun_obj.model_obj(a_tens[None])
res_np = res.detach().numpy()[0][0]
hplotc.ListPlot(res_np)


# Object below is loaded from `model_based_on_measured`
# And `InhomogRemoval.py`
container = gen.__getitem__(index=0)
input_tensor = container['input']
mask_tensor = container['mask']
mask_array = mask_tensor.numpy()
target_tensor = container['target']

input_cpx = input_tensor[::2] + 1j * input_tensor[1::2]
input_sum_of_absolutes = np.abs(input_cpx).mean(axis=0)

modelrun_obj.model_obj.eval()
res = modelrun_obj.model_obj((input_tensor * mask_tensor)[None])
hplotc.ListPlot([input_cpx], augm='np.abs', ax_off=True)

mask_obj = hplotc.MaskCreator(target_tensor.numpy())
hplotc.ListPlot([[target_tensor.numpy()[0], res.detach().numpy()[0][0] * mask_array[0]]], ax_off=True)

hplotc.ListPlot([[input_sum_of_absolutes / target_tensor.numpy()[0], input_sum_of_absolutes / res.detach().numpy()[0][0] * mask_array[0]]], ax_off=True)


import smoothing.L0_gradient_minimization as L0_smooth
res_output = res.detach().numpy()[0][0]
input_sum_of_absolutes = np.abs(input_cpx).sum(axis=0)
lmbd = 0.015  # Lower this.. less detail
beta_max = 1.0e2  # Lower this... more detail
A_smooth = L0_smooth.l0_gradient_minimization_2d(res_output, lmbd, beta_max)
corrected_image = input_sum_of_absolutes / A_smooth
hplotc.ListPlot([[target_tensor.numpy()[0], corrected_image]], ax_off=True)