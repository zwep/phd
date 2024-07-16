
"""
Implement here some code that plots the influence of the adaptive smoothing
"""
import helper.array_transf
import helper.array_transf as harray
import numpy as np
import matplotlib.pyplot as plt
import helper.misc as hmisc
dd_target = '/home/bugger/Documents/paper/inhomogeneity removal/result_models/test_split/test_01/target.png'
dd_uncor = '/home/bugger/Documents/paper/inhomogeneity removal/result_models/test_split/test_01/uncorrected.png'
from PIL import Image
target_array = np.array(Image.open(dd_target))
input_array = np.array(Image.open(dd_uncor))
biasfield_array = input_array / target_array
biasfield_array = helper.array_transf.correct_inf_nan(biasfield_array)

# Load a model...
import objective.inhomog_removal.executor_inhomog_removal as executor
config_path = '/home/bugger/Documents/model_run/inhomog_removal_single_biasf/resnet_15_juli'
config_param = hmisc.convert_remote2local_dict(config_path, path_prefix='/derp')
decision_obj = executor.DecisionMaker(config_file=config_param, debug=True, inference=False, load_model_only=True)  # ==>>
modelrun_obj = decision_obj.decision_maker()
modelrun_obj.load_weights()
import torch
input_array = harray.scale_minmax(input_array)
input_tens = torch.from_numpy(input_array[None])
with torch.no_grad():
    res = modelrun_obj.model_obj(input_tens[None].float())

Z = result_model_np
import helper.plot_class as hplotc
n_points = Z.shape[0]
# He kijk je hebt weer je eigen werk ontkracht....
Z_smooth = harray.smooth_image(Z, n_kernel=4)
Z_adapt_smooth, adapt_smooth_bins = harray.adaptive_smoothing_grid(Z, smoothing_kernel_size=[8, 16, 32])

fig, ax = plt.subplots(2)
ax[0].plot(Z[n_points//2], label='signal')
ax[0].plot(Z_smooth[n_points//2], label='fixed smoothing')
ax[0].plot(Z_adapt_smooth[n_points//2], label='adaptive smoothing')
plt.legend()

ax[1].plot(adapt_smooth_bins[0][n_points//2], label='level 0')
ax[1].plot(adapt_smooth_bins[1][n_points//2], label='level 1')
ax[1].plot(adapt_smooth_bins[2][n_points//2], label='level 2')
plt.legend()