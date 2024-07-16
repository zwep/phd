import helper.array_transf
import helper.array_transf as harray
import objective.inhomog_removal.executor_inhomog_removal as executor
import h5py
import helper.plot_class as hplotc
from skimage.util import img_as_ubyte, img_as_uint
import re
import pydicom
import skimage.transform as sktransform
import os
import helper.misc as hmisc
import numpy as np
import os
import glob

"""
Here we apply the bias field correciton model to the 3T data to see how good it is
"""

# model_path_dir = '/home/bugger/Documents/model_run/inhomog_removal_single_biasf' # ==>
model_path_dir = '/home/bugger/Documents/model_run/inhomog_removal_single' # ==>
# Overview of all the models available in model_path_dir
model_path_list = [os.path.join(model_path_dir, x) for x in os.listdir(model_path_dir)]
model_path_list = [x for x in model_path_list if 'result' not in x]
# Getting the latest directory.... (based on creation time)
model_path_sel = sorted(glob.glob(model_path_dir + '/*'), key=os.path.getctime)
# model_path_sel = [x for x in model_path_sel if '15_juli' in x]
model_path_sel = [x for x in model_path_sel if '07_juli' in x]

"""
Load model
"""
i_model_path = model_path_sel[0]
model_name = os.path.basename(i_model_path)
config_param = hmisc.convert_remote2local_dict(i_model_path, path_prefix='/media/bugger/MyBook/data/semireal')
# Otherwise squeeze will not work properly..
config_param['data']['batch_size'] = 1
decision_obj = executor.DecisionMaker(config_file=config_param, debug=False,
                                      load_model_only=True, inference=True, device='cpu')  # ==>>
modelrun_obj = decision_obj.decision_maker()
target_type = modelrun_obj.config_param['data']['target_type']

modelrun_obj.load_weights()
if modelrun_obj.model_obj:
    modelrun_obj.model_obj.eval()
else:
    modelrun_obj.generator.eval()

"""
Load the 3T data..
Compare it to 1.5T
"""

# This is not the best result.. since they do not include the legs at 3T
ddata_base = '/media/bugger/WORK_USB/astrid_umc'
# ddata_base = '/media/bugger/MyBook/data/3T_scan/astrid_umc'
ddata_1p5t = os.path.join(ddata_base, 'input/36_MR.h5')
ddata_3t = os.path.join(ddata_base, 'target/36_MR.h5')
dest_dir = os.path.join(ddata_base, 'corrected')
file_name = os.path.basename(ddata_1p5t)
# dest_file = os.path.join(dest_dir, 'direct_model_' + file_name)
dest_file = os.path.join(dest_dir, file_name)


with h5py.File(ddata_1p5t, 'r') as f:
    A_1p5T = np.array(f['data'])

with h5py.File(ddata_3t, 'r') as f:
    A_3T = np.array(f['data'])

print(A_3T.shape[0])
print(A_1p5T.shape[0])

A_1p5T = harray.scale_minmax(A_1p5T, axis=(-2, -1))
A_3T = harray.scale_minmax(A_3T, axis=(-2, -1))
# hplotc.SlidingPlot(A_1p5T[::3])
# hplotc.SlidingPlot(A_3T)

A_corrected = []
import torch
for i_slice_1p5t, i_slice_3t in zip(A_1p5T, A_3T):
    A_tensor = torch.as_tensor(i_slice_3t[np.newaxis, np.newaxis]).float()

    with torch.no_grad():
        res = modelrun_obj.model_obj(A_tensor)

    bias_field = res.numpy()[0][0]
    bias_field_smoothed_adaptive = harray.smooth_image(bias_field, n_kernel=bias_field.shape[0] // 64, conv_boundary='wrap')
    corrected_image = i_slice_3t / bias_field_smoothed_adaptive
    corrected_image = helper.array_transf.correct_inf_nan(corrected_image)
    corrected_image = harray.scale_minmax(corrected_image)
    A_corrected.append(corrected_image)

with h5py.File(dest_file, 'w') as h5_obj:
    h5_obj.create_dataset('data', data=np.array(A_corrected))

"""
Now compare stuff...
"""

from data_prep.objective.prostate_weighting.extract_MRI_images import CompareImages
compare_obj = CompareImages(ddata_1p5t, ddata_3t)
# compare_score = compare_obj.compare_mask_array()
compare_score = compare_obj.compare_array()
print(compare_score)

compare_obj2 = CompareImages(ddata_1p5t, dest_file)
# compare_score2 = compare_obj2.compare_mask_array()
compare_score2 = compare_obj2.compare_array()
print(compare_score2)

hplotc.SlidingPlot(compare_obj.array_1)
hplotc.SlidingPlot(compare_obj.array_2)
hplotc.SlidingPlot(compare_obj2.array_2)
hplotc.ListPlot([compare_obj.array_2[5], compare_obj2.array_2[5]], subtitle=[['3T'], ['cor 3T']])
hplotc.ListPlot([compare_obj.array_2[50], compare_obj2.array_2[50]], subtitle=[['3T'], ['cor 3T']])
hplotc.ListPlot([compare_obj.array_2[140], compare_obj2.array_2[140]], subtitle=[['3T'], ['cor 3T']])