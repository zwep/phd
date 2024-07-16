import h5py
import torch
import helper.plot_class as hplotc
import helper.array_transf as harray
import numpy as np
import os
from skimage import img_as_uint
import helper.misc as hmisc
import objective.prostate_weighting.recall_prostate_weighting as recall_model


"""
Here we create the by DL model processed results

Meaning that we've already done the bias field correction
"""

ddata_base = '/home/bugger/Documents/data/3T/prostate/prostate_weighting/test'
ddata_1p5T = os.path.join(ddata_base, 'input')
ddata_mask = os.path.join(ddata_base, 'mask')
ddata_3T = os.path.join(ddata_base, 'target')
ddata_3T_cor = os.path.join(ddata_base, 'target_corrected')

ddata_3T_cor_regular = os.path.join(ddata_base, 'target_regular_corrected')
ddata_3T_cor_gan = os.path.join(ddata_base, 'target_gan_corrected')
if not os.path.isdir(ddata_3T_cor_regular):
    os.makedirs(ddata_3T_cor_regular)

if not os.path.isdir(ddata_3T_cor_gan):
    os.makedirs(ddata_3T_cor_gan)

"""
Load the models
"""

# Load the local config file
dconfig = '/home/bugger/Documents/model_run/prostate_weighting/regular/nov_run'
config_file = hmisc.convert_remote2local_dict(dconfig, path_prefix='', name='config_param.json')

# Load the GAN config file
dconfig_gan = '/home/bugger/Documents/model_run/prostate_weighting/gan/jan_run'
config_file_gan = hmisc.convert_remote2local_dict(dconfig_gan, path_prefix='', name='config_param.json')

recall_obj_regular = recall_model.RecallProstateWeighting()
model_obj_regular = recall_obj_regular.get_model_object(config_file=config_file)

recall_obj_gan = recall_model.RecallProstateWeighting(load_discriminator=False)
config_file_gan['model']['config_gan']['reload_weights_generator_config'] = {"status": False}
model_obj_gan = recall_obj_gan.get_model_object(config_file=config_file_gan)


file_list = os.listdir(ddata_3T)
for sel_file in file_list:
    print("Processing file ", sel_file)
    file_name, ext = os.path.splitext(sel_file)
    file_3T_mask = os.path.join(ddata_mask, file_name + '_target' + ext)
    file_3T_cor = os.path.join(ddata_3T_cor, sel_file)

    ddest_regular = os.path.join(ddata_3T_cor_regular, sel_file)
    ddest_gan = os.path.join(ddata_3T_cor_gan, sel_file)

    with h5py.File(file_3T_cor, 'r') as f:
        array_3T = np.array(f['data'])

    with h5py.File(file_3T_mask, 'r') as f:
        mask_array_3T = np.array(f['data'])

    corrected_3T_regular = []
    corrected_3T_gan = []
    for i_slice, i_mask in zip(array_3T, mask_array_3T):
        array_3T_cor_tens = torch.from_numpy(i_slice * i_mask)[None, None].float()
        with torch.no_grad():
            res_regular = model_obj_regular.model_obj(array_3T_cor_tens)

        with torch.no_grad():
            res_gan = model_obj_gan.generator(array_3T_cor_tens)

        res_regular_np = res_regular.numpy()[0][0]
        res_gan_np = res_gan.numpy()[0][0]

        corrected_3T_regular.append(res_regular_np)
        corrected_3T_gan.append(res_gan_np)

    temp_gan = img_as_uint(harray.scale_minmax(np.array(corrected_3T_gan))) * mask_array_3T
    temp_regular = img_as_uint(harray.scale_minmax(np.array(corrected_3T_regular))) * mask_array_3T

    # hplotc.SlidingPlot(temp_gan)
    # hplotc.SlidingPlot(temp_regular)

    with h5py.File(ddest_gan, 'w') as f:
        f.create_dataset('data', data=temp_gan)

    with h5py.File(ddest_regular, 'w') as f:
        f.create_dataset('data', data=temp_regular)