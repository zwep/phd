

"""
Load model
"""

import objective.shim_prediction.executor_shim_prediction as executor  # ==>
import helper.misc as hmisc

model_path = '/home/bugger/Documents/model_run/shim_prediction_model/shimnet'
# model_path = '/home/bugger/Documents/model_run/shim_prediction_model/piennet_max_value'
# model_path = '/home/bugger/Documents/model_run/shim_prediction_model/piennet_abs_comparisson'

config_param = hmisc.convert_remote2local_dict(model_path, path_prefix='/media/bugger/MyBook/data/semireal')
# Otherwise squeeze will not work properly..
config_param['data']['batch_size'] = 1
# We now give an index GPU such that ALL the models will put their information on ONE gpu.
# This should be the new way.. to distinguish between GANs and non GANs
decision_obj = executor.DecisionMakerShimPredicton(config_file=config_param, load_model_only=True, inference=True, device='cpu')  # ==>>
modelrun_obj = decision_obj.decision_maker()
modelrun_obj.load_weights()


"""
Load the new data..
"""

import os
import reconstruction.ReadCpx as read_cpx

# ddata = '/media/bugger/WORK_USB/2021_06_21/te_21515'
ddata = '/media/bugger/WORK_USB/newsession'
b1_shim_file = [x for x in os.listdir(ddata) if 'shim' in x and x.endswith('cpx')]
if len(b1_shim_file) == 1:
    b1_shim_file = b1_shim_file[0]
    b1_shim_file = os.path.join(ddata, b1_shim_file)
else:
    print('Found multiple shim files ', b1_shim_file)

img_obj = read_cpx.ReadCpx(b1_shim_file)
img_array = img_obj.get_cpx_img()

"""
Prep data for inference
"""

import numpy as np
import torch
import helper.array_transf as harray

new_size = (8, 8, 128, 128)
input_array = np.squeeze(img_array)
temp_array = harray.resize_complex_array(input_array, new_shape=new_size, preserve_range=True)
ny, nx = temp_array.shape[-2:]
new_shape = (-1, ny, nx)
input_real = temp_array.real.reshape(new_shape)
input_imag = temp_array.imag.reshape(new_shape)
input_array = np.concatenate([input_real, input_imag])
input_array = input_array / np.max(input_array)
input_tensor = torch.from_numpy(input_array).float()

with torch.no_grad():
    result_phase = modelrun_obj.model_obj(input_tensor[None])
transmit_phase = result_phase[:, :8]
receive_phase = result_phase[:, 8:]

print('transmit phase settings')
print([x / torch.max(transmit_phase) * 360 for x in transmit_phase])
print('receive phase settings')
print([x / torch.max(receive_phase) * 360 for x in receive_phase])

"""
See result of prediction....
"""

from helper_torch.loss import ShimLoss
loss_obj = ShimLoss()
real_input, imag_input = loss_obj.process_input(input_tensor[None])
tx_input_real, tx_input_imag = loss_obj.apply_transmit_phase(real_input, imag_input, phase_prediction=transmit_phase)
# On top of that, apply the receive phase setting
rx_tx_input_real, rx_tx_input_imag = loss_obj.apply_receive_phase(tx_input_real, tx_input_imag, phase_prediction=receive_phase)

# Now calculate the magnitude. We have succesfully avoided any complex arguments
real_sum = torch.sum(torch.sum(rx_tx_input_real, dim=1), dim=1)
imag_sum = torch.sum(torch.sum(rx_tx_input_imag, dim=1), dim=1)
result = torch.sqrt(real_sum ** 2 + imag_sum ** 2)
import helper.plot_class as hplotc
hplotc.SlidingPlot(result.detach().numpy(), title=os.path.basename(model_path))


"""
"""