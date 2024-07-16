import helper.plot_class as hplotc
import os
import torch
from model.FBPConvnet import FBPCONVNet
import numpy as np
import objective_helper.reconstruction as obj_helper
from objective_configuration.reconstruction import DPHANTOM_input, DPHANTOM_target, DPLOT

dplot_undersampled_phantom = os.path.join(DPLOT, 'FBPconvnet_undersampled_phantom.png')

"""
Load data phantom
"""

n_phantom = 1
input_array = np.load(DPHANTOM_input)
phantom_tensor = torch.from_numpy(input_array).float()[:n_phantom, None]
target_array = np.load(DPHANTOM_target)
sel_target_array = target_array[:n_phantom]

"""
Load model
"""

# Create model
model_obj = FBPCONVNet()
recovered_state_dict = obj_helper.get_mapping_fbpconvnet(num_batch=1)
model_obj.load_state_dict(recovered_state_dict)
# model_obj.eval()
with torch.no_grad():
    res_phantom_tensor = model_obj(phantom_tensor)


# Visualize phantom array
fig_obj = hplotc.ListPlot([phantom_tensor, res_phantom_tensor, sel_target_array], subtitle=[['input']*n_phantom, ['result']*n_phantom, ['target']*n_phantom])
fig_obj.figure.savefig(dplot_undersampled_phantom)
