
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

import objective.undersampled_recon.executor_undersampled_recon as executor_radial


# model_path = '/home/bugger/Documents/model_run/undersampled_recon/resnet_24_sep'
model_path = '/data/seb/model_run/undersampled_recon_radial_20/config_00'
# config_param = hmisc.convert_remote2local_dict(model_path, path_prefix='/data/seb/unfolded_radial')
# config_param['data']['batch_size'] = 1
decision_obj = executor_radial.DecisionMakerRecon(model_path=model_path, debug=True, load_model_only=False, inference=True, device='cpu')  # ==>>
decision_obj.config_param['data']['batch_size'] = 1
modelrun_obj = decision_obj.decision_maker()
modelrun_obj.load_weights()
if modelrun_obj.model_obj:
    modelrun_obj.model_obj.eval()
else:
    modelrun_obj.generator.eval()


for i_dataset in range(modelrun_obj.test_loader.dataset.n_datasets):
    modelrun_obj.test_loader.selected_dataset = i_dataset
    modelrun_obj.store_test_predictions(i_dataset)
