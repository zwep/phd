
import os
import numpy as np
import torch
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.misc as hmisc
import objective.undersampled_recon.executor_undersampled_recon as executor
import helper.array_transf as harray
import scipy.io
from skimage.util import img_as_ubyte, img_as_uint
import skimage.transform as sktransf
import imageio
import reconstruction.ReadCpx as read_cpx
import reconstruction.ReadRec as read_rec


"""
Try to see if we can fix the FOV thing..

Doing so by parameters is really strange and weird..

"""

patient_dir = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_05_02/V9_19531'

# Cpx file of carteisna one is of course... SENSE'd
groundtruth_file = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac/V9_19531/v9_02052021_1430230_5_2_cine1slicer2_traV4.mat'
# No CPX so that we can get the FOV etc. from the param file
cartesian_file = os.path.join(patient_dir, 'v9_02052021_1430230_5_3_cine1slicer2_traV4')
radial_fs_file = os.path.join(patient_dir, 'v9_02052021_1431531_6_3_transradialfastV4')
radial_us_file = os.path.join(patient_dir, 'v9_02052021_1432554_7_3_transradialfast_high_timeV4')
radial_no_trigger_file = os.path.join(patient_dir, 'v9_02052021_1434042_8_3_transradial_no_trigV4')

# First display the par-file FOV stuff
cart_obj = read_cpx.ReadCpx(cartesian_file)
cart_param = cart_obj.get_par_file()
# Get the array via the .mat object...
cart_array = scipy.io.loadmat(groundtruth_file)['reconstructed_data']
cart_array = np.moveaxis(np.squeeze(np.abs(cart_array)), -1, 0)[0]
print('Cartesian shape', cart_array.shape, cart_param['FOV (ap,fh,rl) [mm]'])

radial_us_obj = read_cpx.ReadCpx(radial_us_file)
radial_us_param = radial_us_obj.get_par_file()

radial_us_array = radial_us_obj.get_cpx_img()
radial_us_array_summed = np.squeeze(np.abs(radial_us_array)).sum(axis=0)[0][::-1, ::-1]
print('Radial us shape', radial_us_array_summed.shape, radial_us_param['FOV (ap,fh,rl) [mm]'])

hplotc.ListPlot([radial_us_array_summed, cart_array])

"""
Load model to see if we can create a work-around with masks
"""

import objective.undersampled_recon.executor_undersampled_recon as executor_radial

# model_path = '/home/bugger/Documents/model_run/various_us_recon/resnet_radial_20'
model_path = '/home/bugger/Documents/model_run/undersampled_recon/resnet_24_sep'
config_param = hmisc.convert_remote2local_dict(model_path, path_prefix='/media/bugger/MyBook/data/semireal')
config_param['data']['batch_size'] = 1
decision_obj = executor_radial.DecisionMakerRecon(config_file=config_param, debug=True, load_model_only=True, inference=False, device='cpu')  # ==>>
modelrun_obj = decision_obj.decision_maker()
modelrun_obj.load_weights()
if modelrun_obj.model_obj:
    modelrun_obj.model_obj.eval()
else:
    modelrun_obj.generator.eval()

import helper.misc as hmisc
input_tens = hmisc.convert2tensor_complex(np.squeeze(radial_us_array)[-8:, 0])
with torch.no_grad():
    res = modelrun_obj.model_obj(input_tens)[0][0].numpy()[::-1, ::-1]

res = sktransf.resize(res, (512, 512), anti_aliasing=False)
cart_array = sktransf.resize(cart_array, (512, 512), anti_aliasing=False)

res_mask = harray.get_treshold_label_mask(res)
res_mask, _ = harray.convex_hull_image(res_mask)
cart_mask = harray.get_treshold_label_mask(cart_array)
cart_mask, _ = harray.convex_hull_image(cart_mask)
hplotc.ListPlot([res_mask, res, cart_mask, cart_array])

res_index0, res_index1 = np.argwhere(np.diff(res_mask.sum(axis=1)>0))
cart_index0, cart_index1 = np.argwhere(np.diff(cart_mask.sum(axis=1)>0))
cart_width = cart_index1 - cart_index0
res_width = res_index1 - res_index0
n_width = res_mask.shape[0]
ratio = cart_width / res_width
new_width = int(n_width * ratio)
res = sktransf.resize(res, (new_width, new_width), anti_aliasing=False)
res_new_size = np.zeros(cart_array.shape)
res_new_size[n_width//2 - new_width//2: n_width//2 + new_width//2 +1, n_width//2 - new_width//2: n_width//2 + new_width//2+1] = res

res_new_mask = harray.get_treshold_label_mask(res_new_size)
res_new_mask, _ = harray.convex_hull_image(res_new_mask)

res_new_pos, res_mask_new_pos = harray.get_center_transformation(res_new_size, x_mask=res_new_mask)
cart_new_pos, cart_mask_new_pos = harray.get_center_transformation(cart_array, x_mask=cart_mask)

cart_new_pos = harray.scale_minmax(cart_new_pos)
hplotc.ListPlot([cart_new_pos * cart_mask_new_pos - res_new_pos * res_mask_new_pos])