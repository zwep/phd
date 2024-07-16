import os
import numpy as np
import sys

# Self created code
import helper.misc as hmisc
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.nvidia_parser as hnvidia
import helper.model_setting as hmodel_set
import objective.super_resolution.executor_super_resolution as executor  # ==>
import json
## Load config
model_path = "/home/bugger/Documents/model_run/SR_test"

config_file_name = "config_param.json"  # ==>>
config_file = os.path.join(model_path, config_file_name)

with open(config_file, 'r') as f:
    text_obj = f.read()
    config_model = json.loads(text_obj)

## Load model
debug_ind = True
index_gpu, p_gpu = hnvidia.get_free_gpu_id(claim_memory=config_model['gpu_frac'])
decision_obj = executor.DecisionMakerSuperRes(config_file=config_model, debug=debug_ind, index_gpu=index_gpu,
                                              load_model_only=True)  # ==>>
modelrun_obj = decision_obj.decision_maker()

## Load data

from PIL import Image
import torch
import helper.array_transf as harray
image_path = os.path.join(model_path, "Untitled.png")
A = np.array(Image.open(image_path).convert('LA'))[:, :, 0]
A = harray.scale_minmax(A)
A = torch.as_tensor(A).float()

res = modelrun_obj.model_obj(A[None, None]).detach().numpy()[0, 0]
res = harray.smooth_image(res)
res = harray.scale_minpercentile(res, q=80)
np.abs(res).max()
res.min()
np.abs(res).min()
res[res<0.5] = 0
hplotf.plot_3d_list(res, augm='np.abs')