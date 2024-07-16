"""
We can hard code the pretrained model paths

"""

import os
from objective_configuration.reconstruction import DPRETRAINED

subdir_list = os.listdir(DPRETRAINED)
list_of_pretrained_model = []
for i_subdir in subdir_list:
    model_path = os.path.join(DPRETRAINED, i_subdir)
    if os.path.isdir(model_path):
        list_files = os.listdir(model_path)
        filter_f = sorted([x for x in list_files if x.endswith('pt')])
        print('Available .pt files ', filter_f)
        print('Selecting .pt file ', filter_f[0])
        sel_file = os.path.join(model_path, filter_f[0])
        list_of_pretrained_model.append(sel_file)

for i_file in list_of_pretrained_model:
    print(i_file)

"""
Result..

and this should help somehow

/home/sharreve/local_scratch/mri_data/pretrained_networks/direct/varnet_seb/model_3000.pt
/home/sharreve/local_scratch/mri_data/pretrained_networks/direct/base_conjgradnet/model_55500.pt
/home/sharreve/local_scratch/mri_data/pretrained_networks/direct/rim_seb/model_63000.pt
/home/sharreve/local_scratch/mri_data/pretrained_networks/direct/multidomainnet/model_50000.pt
/home/sharreve/local_scratch/mri_data/pretrained_networks/direct/xpdnet_seb/model_14000.pt
/home/sharreve/local_scratch/mri_data/pretrained_networks/direct/recurrentvarnet/model_107000.pt
/home/sharreve/local_scratch/mri_data/pretrained_networks/direct/jointicnet/model_42500.pt
/home/sharreve/local_scratch/mri_data/pretrained_networks/direct/kikinet/model_44500.pt
/home/sharreve/local_scratch/mri_data/pretrained_networks/direct/base_iterdualnet/model_33500.pt
/home/sharreve/local_scratch/mri_data/pretrained_networks/direct/unet_seb/model_10000.pt
/home/sharreve/local_scratch/mri_data/pretrained_networks/direct/lpdnet/model_96000.pt

"""