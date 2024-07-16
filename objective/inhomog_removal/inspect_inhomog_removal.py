# encoding: utf-8

import getpass
import os
import sys

import numpy as np
import helper.array_transf as htransf
import skimage.transform as sktransf
import torch.nn.functional as F
import helper.plot_class as hplotc
import numpy as np
import json
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.misc as hmisc
import helper_torch.misc as htmisc
import helper.model_setting as hmodel_set
import torch.utils.data
import objective.inhomog_removal.executor_inhomog_removal as executor
import functools

"""
Load model with config param.
"""
config_path = '/home/bugger/Documents/model_run/inhom_biasfield'
config_path = '/home/bugger/Documents/model_run/inhom_removal_4ch'
path_prefix_prostate = '/home/bugger/Documents/data/semireal'
config_param = hmisc.convert_remote2local_dict(config_path, path_prefix=path_prefix_prostate)
decision_obj = executor.DecisionMaker(config_file=config_param, debug=True, inference=True, load_model_only=True)  # ==>>
modelrun_obj = decision_obj.decision_maker()
modelrun_obj.load_weights()

"""
Load some local data
"""
import reconstruction.ReadCpx as read_cpx
plot_intermediate_results = False

cardiac_4ch_file = '/media/bugger/MyBook/data/7T_data/cartesian_radial_dataset_4ch/test/target/v9_02052021_1448567.npy'
A = np.load(cardiac_4ch_file)[10]
A_tensor = torch.from_numpy(np.abs(A[None, None][:, :, ::-1])).float()

with torch.no_grad():
    res = modelrun_obj.model_obj(A_tensor)

hplotc.ListPlot([res, A_tensor, A_tensor / res])
