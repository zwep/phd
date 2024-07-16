import objective.reconstruction.executor_reconstruction as executor
from skimage.metrics import structural_similarity
import objective.reconstruction.postproc_reconstruction as postproc_reconstruction
from objective_configuration.reconstruction import DINFERENCE
import helper.plot_class as hplotc
import torch
import numpy as np
import os
import helper.array_transf as harray
import helper.misc as hmisc

"""
We have also trained a model our selves... with my own package and such

lets see how this one does it one the inference dataset
"""

# How to reaload a model again...

model_path = '/local_scratch/sharreve/model_run/recon_test/config_00'
data_dir = os.path.join(DINFERENCE, 'input')
target_dir = os.path.join(DINFERENCE, 'target')
storage_dir = '/home/sharreve'

postproc_obj = postproc_reconstruction.PostProcReconstruction(image_dir=target_dir,
                                                       target_dir=storage_dir,
                                                       config_path=model_path,
                                                       executor_module=executor, config_name='config_param.json')

def load_prep_data(filename):
    A_mid = hmisc.load_array(filename, data_key='kspace', sel_slice='mid')
    A_cpx = A_mid[..., ::2] + 1j * A_mid[..., 1::2]
    A_cpx = np.moveaxis(A_cpx, -1, 0)
    A_cpx = A_cpx[-8:]
    A_cpx = np.fft.fft2(A_cpx)
    A_perc_99 = np.percentile(np.abs(A_cpx), 99)
    A_cpx = A_cpx / A_perc_99
    A_cpx = A_cpx[:, ::-1, ::-1]
    A_split = np.concatenate([A_cpx.real, A_cpx.imag], 0)
    A_tens = torch.from_numpy(A_split[None]).float()
    return A_cpx, A_tens

selected_ref_model = '/local_scratch/sharreve/paper/reconstruction/inference/unet_RADIAL/25p/train_mixed/undersampled/'

file_list = os.listdir(data_dir)
file_list = [x for x in file_list if x.endswith('h5')]
for i_file in file_list:
    print()
    basename = hmisc.get_base_name(i_file)
    A = os.path.join(data_dir, i_file)
    B = os.path.join(target_dir, i_file)
    A_cpx, A_tens = load_prep_data(A)
    B_cpx, B_tens = load_prep_data(B)
    A_tens = A_tens.to(postproc_obj.modelrun_obj.device)
    with torch.no_grad():
        res = postproc_obj.modelrun_obj.model_obj(A_tens)
    #
    other_model_file = os.path.join(selected_ref_model, i_file)
    other_model = np.abs(hmisc.load_array(other_model_file, data_key='reconstruction', sel_slice='mid'))
    other_model = harray.scale_minmax(other_model)
    A_sos = np.sqrt((np.abs(A_cpx) ** 2).sum(axis=0))
    B_sos = np.sqrt((np.abs(B_cpx) ** 2).sum(axis=0))
    x = harray.scale_minmax(A_sos)
    y = harray.scale_minmax(B_sos)
    z = np.squeeze(res.cpu().numpy())
    # Expand on location..
    # Perofrm difference minmax scaling..?
    ssim_value = structural_similarity(y, z, data_range=1)
    print('my model', ssim_value)
    ssim_value = structural_similarity(y, other_model, data_range=1)
    print('ref model', ssim_value)
    fig_obj = hplotc.ListPlot([A_sos, res.cpu().numpy(), B_sos])
    fig_obj.figure.savefig(f'/home/sharreve/{basename}.png')


