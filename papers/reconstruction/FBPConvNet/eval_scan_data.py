import helper.plot_class as hplotc
import os
import helper.array_transf as harray
import torch
from model.FBPConvnet import FBPCONVNet
import numpy as np
import objective_helper.reconstruction as obj_helper
from objective_configuration.reconstruction import DSCAN_cartesian, DPLOT, DSCAN_us_radial, DSCAN_us_spokes

dplot_undersample_scan = os.path.join(DPLOT, 'FBPconvnet_undersampled_scan.png')

"""
Load cardiac data
"""

# Load data and undersample
sel_cpx_array = np.load(DSCAN_cartesian)
traj = obj_helper.undersample_trajectory(sel_cpx_array.shape)
radial_undersampled_image = obj_helper.undersample_img(sel_cpx_array,  traj)

# Taking the absolute value of the image to avoid destructive interference patterns
abs_image = np.abs(radial_undersampled_image)
abs_image = harray.scale_minmax(abs_image)
undersampled_card_tensor = torch.from_numpy(abs_image).float()

"""
Load model
"""

# Create model
model_obj = FBPCONVNet()
recovered_state_dict = obj_helper.get_mapping_fbpconvnet(num_batch=1)
model_obj.load_state_dict(recovered_state_dict)
# The 'eval' mode messes stuff up
model_obj.eval()

with torch.no_grad():
    res_undersampled_card = model_obj(undersampled_card_tensor[None])


# Visualize undersampling
fig_obj = hplotc.ListPlot([sel_cpx_array, res_undersampled_card, radial_undersampled_image], augm='np.abs', subtitle=[['orig'], ['result'], ['target']],
                          col_row=(3, 1))
fig_obj.figure.savefig(dplot_undersample_scan)
