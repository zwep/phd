"""
Model resultaten laten zien....
"""

import matplotlib.pyplot as plt
import os
import pydicom
import h5py
import pydicom
import scipy.io
import numpy as np

import helper.array_transf
import helper.plot_class as hplotc
import helper.array_transf as harray

remote_location = '/local_scratch/sharreve/paper/inhomog_removal'
local_location = '/home/bugger/Documents/paper/inhomogeneity removal'
local_location_input = os.path.join(local_location, 'data_creation')
local_location_input_per_coil = os.path.join(local_location_input, 'data_per_coil')


# Plaatje 7
# Visualizatie van model resultaten one 2 one op Daan zn data

ddata = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Seb_pred/7TMRI008'
dir_one2one_expansion = os.path.join(ddata, 'corrected_expansion.dcm')
dir_one2one_biasfield = os.path.join(ddata, 'corrected_biasfield.dcm')
dir_one2one_rho = os.path.join(ddata, 'corrected_rho.dcm')
dir_one2one_original = os.path.join(ddata, 'uncorrected_rho.dcm')

one2one_expansion = pydicom.read_file(dir_one2one_expansion).pixel_array
one2one_biasfield = pydicom.read_file(dir_one2one_biasfield).pixel_array
one2one_rho = pydicom.read_file(dir_one2one_rho).pixel_array
one2one_uncorrected = pydicom.read_file(dir_one2one_original).pixel_array
n_slice = one2one_biasfield.shape[0]

import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 18
plot_obj = hplotc.ListPlot([[one2one_uncorrected[n_slice // 2], one2one_rho[n_slice // 2], one2one_biasfield[n_slice // 2], one2one_expansion[n_slice // 2]]], ax_off=True,
                vmin=(0, 65364), subtitle=[['a)', 'b)', 'c)', 'd)']])
plot_obj.figure.savefig(os.path.join(local_location, 'result_one2one_daan.png'), bbox_inches='tight')


# Visualizatie van model one 2 one op Volunteer
ddata = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/prediction/pr_06012021_1647041_12_3_t2wV4'
dir_one2one_expansion = os.path.join(ddata, 'corrected_expansion.dcm')
dir_one2one_biasfield = os.path.join(ddata, 'corrected_biasfield.dcm')
dir_one2one_rho = os.path.join(ddata, 'corrected_rho.dcm')
dir_one2one_original = os.path.join(ddata, 'uncorrected_rho.dcm')

one2one_expansion = pydicom.read_file(dir_one2one_expansion).pixel_array
one2one_rho = pydicom.read_file(dir_one2one_rho).pixel_array
one2one_biasfield = pydicom.read_file(dir_one2one_biasfield).pixel_array
one2one_uncorrected = pydicom.read_file(dir_one2one_original).pixel_array

plot_obj = hplotc.ListPlot([[one2one_uncorrected, one2one_rho, one2one_biasfield, one2one_expansion]], ax_off=True,
                vmin=(0, 2**16), subtitle=[['a)', 'b)', 'c)', 'd)']])
plot_obj.figure.savefig(os.path.join(local_location, 'result_one2one_volunteer.png'), bbox_inches='tight')


# Visualizatie van model one 2 one op Bart zn data
ddata = '/media/bugger/MyBook/data/multiT_scan/prostaat/prediction_model/V509900001'
dir_one2one_biasfield = os.path.join(ddata, 'corrected_biasfield.dcm')
dir_one2one_rho = os.path.join(ddata, 'corrected_rho.dcm')
dir_one2one_original_7T = os.path.join(ddata, 'uncorrected_7T.dcm')
dir_one2one_original_3T = os.path.join(ddata, 'uncorrected_3T.dcm')
dir_one2one_expansion = os.path.join(ddata, 'corrected_expansion.dcm')

one2one_expansion = pydicom.read_file(dir_one2one_expansion).pixel_array
one2one_rho = pydicom.read_file(dir_one2one_rho).pixel_array
one2one_biasfield = pydicom.read_file(dir_one2one_biasfield).pixel_array
one2one_uncorrected_3T = pydicom.read_file(dir_one2one_original_3T).pixel_array
one2one_uncorrected_7T = pydicom.read_file(dir_one2one_original_7T).pixel_array

plot_obj = hplotc.ListPlot([[one2one_uncorrected_7T, one2one_rho, one2one_biasfield, one2one_expansion, one2one_uncorrected_3T]],
                           ax_off=True, vmin=(0, 2**16), subtitle=[['a)', 'b)', 'c)', 'd)', 'e)']])
plot_obj.figure.savefig(os.path.join(local_location, 'result_one2one_bart.png'), bbox_inches='tight')

# Visualizatie van model 8 to one volunteer
ddata = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/prediction_multi/pr_06012021_1647041_12_3_t2wV4'
dir_one2one_expansion = os.path.join(ddata, 'corrected_expansion.dcm')
dir_one2one_biasfield = os.path.join(ddata, 'corrected_biasfield.dcm')
dir_one2one_rho = os.path.join(ddata, 'corrected_rho.dcm')
dir_one2one_original = os.path.join(ddata, 'uncorrected.dcm')

one2one_rho = pydicom.read_file(dir_one2one_rho).pixel_array
one2one_biasfield = pydicom.read_file(dir_one2one_biasfield).pixel_array
one2one_expansion = pydicom.read_file(dir_one2one_expansion).pixel_array
one2one_uncorrected = pydicom.read_file(dir_one2one_original).pixel_array

plot_obj = hplotc.ListPlot([[one2one_uncorrected, one2one_rho, one2one_biasfield, one2one_expansion]], ax_off=True,
                vmin=(0, 2**16), subtitle=[['a)', 'b)', 'c)', 'd)']])
plot_obj.figure.savefig(os.path.join(local_location, 'result_multi2one_volunteer.png'), bbox_inches='tight')


# Visualizatie van model 8 to 1 op test split

import numpy as np
import torch
import objective.inhomog_removal.executor_inhomog_removal as executor
import helper.misc as hmisc

dest_dir = '/home/bugger/Documents/paper/inhomogeneity removal/result_models'
sel_model_path_dir = '/home/bugger/Documents/model_run/inhomog_removal/resnet_09_juli'
ddata = '/home/bugger/Documents/paper/inhomogeneity removal/data_creation/example_input_array_cpx.npy'

# Load input data
x_input = np.load(ddata)
n_y, n_x = x_input.shape[-2:]
x_inputed = harray.to_stacked(x_input, cpx_type='cartesian', stack_ax=0)
x_inputed = x_inputed.T.reshape((n_x, n_y, -1)).T
A_tensor = torch.as_tensor(x_inputed[np.newaxis]).float()
input_sum_of_absolutes = np.abs(x_input).sum(axis=0)

mask_obj = hplotc.MaskCreator(x_input)
body_mask_array = mask_obj.mask
# Load model and stuff....
config_param = hmisc.convert_remote2local_dict(sel_model_path_dir, path_prefix='/media/bugger/MyBook/data/semireal')
# Otherwise squeeze will not work properly..
config_param['data']['batch_size'] = 1
decision_obj = executor.DecisionMaker(config_file=config_param, debug=False,
                                      load_model_only=True, inference=True, device='cpu')  # ==>>
modelrun_obj = decision_obj.decision_maker()
modelrun_obj.load_weights()
if modelrun_obj.model_obj:
    modelrun_obj.model_obj.eval()

with torch.no_grad():
    output = modelrun_obj.model_obj(A_tensor)

output_abs = output.numpy()[0][0] * body_mask_array
corrected_image = output_abs
bias_field = input_sum_of_absolutes / corrected_image * body_mask_array
bias_field = helper.array_transf.correct_inf_nan(bias_field)
bias_field = harray.scale_minpercentile_both(bias_field, 95)
bias_field[np.abs(bias_field) > 1] = 1
bias_field_smoothed = harray.smooth_image(bias_field, 16)

corrected_image = input_sum_of_absolutes / bias_field_smoothed
corrected_image = helper.array_transf.correct_inf_nan(corrected_image)
corrected_image[np.abs(corrected_image) > 1] = 1

plot_obj = hplotc.ListPlot(output_abs, ax_off=True)
plot_obj.figure.savefig(os.path.join(dest_dir, 'output_direct_model.png'), bbox_inches='tight')

plot_obj = hplotc.ListPlot(bias_field, ax_off=True)
plot_obj.figure.savefig(os.path.join(dest_dir, 'output_direct_model_biasfield.png'), bbox_inches='tight')

plot_obj = hplotc.ListPlot(bias_field_smoothed, ax_off=True)
plot_obj.figure.savefig(os.path.join(dest_dir, 'output_direct_model_biasfield_smoothed.png'), bbox_inches='tight')

plot_obj = hplotc.ListPlot(corrected_image, ax_off=True)
plot_obj.figure.savefig(os.path.join(dest_dir, 'output_direct_model_corrected.png'), bbox_inches='tight')