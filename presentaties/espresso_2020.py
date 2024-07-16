"""

Hier maken we alle plaatjes voor de Espresso meeting
"""

import tooling.shimming.b1shimming_single as mb1
import helper.array_transf as harray
import helper.plot_class as hplotc
import nrrd

import numpy as np


def apply_shim(x, n_chan=8, amp_phase=None):
        # Amp_phase can be a tuple containing.. (amp, phase)
        if amp_phase is None:
            amp = np.ones(n_chan)
            phase = np.random.normal(0, 0.5 * np.sqrt(np.pi), size=n_chan)
        else:
            amp, phase = amp_phase

        cpx_shim = np.array([r * np.exp(1j * phi ) for r, phi in zip(amp, phase)])
        x = np.einsum("tmn, t -> mn", x, cpx_shim)
        return x


def scale_signal_model(x, flip_angle=np.pi/2):
    # Input should an already be shimmed b1 plus image...
    n_y, n_x = x.shape
    y_center, x_center = (n_y // 2, n_x // 2)
    delta_x = int(0.1 * n_y)
    x_sub = x[y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x]
    x_mean = np.abs(x_sub.mean())
    # Taking the absolute values to make sure that values are between 0..1
    # B1 plus interference by complex sum. Then using abs value to scale
    target_angle = np.random.uniform(flip_angle - np.pi/18, flip_angle + np.pi/18)
    x_scaled = np.sin(np.abs(x) / x_mean * target_angle) ** 3
    x_scaled_cpx = x_scaled * np.exp(1j * np.angle(x))
    return x_scaled_cpx


"""
Load very specific paths..
"""

b1m_path = "/home/bugger/Documents/presentaties/Espresso/november2020/M10_to_000029_00001_006__38_inp.npy"
# b1m_path = "/home/bugger/Documents/presentaties/Espresso/november2020/M10_to_000029_00001_006__0_inp.npy"
b1p_path = "/home/bugger/Documents/presentaties/Espresso/november2020/M10_to_000029_00001_006__38_tgt.npy"
# b1p_path = "/home/bugger/Documents/presentaties/Espresso/november2020/M10_to_000029_00001_006__0_tgt.npy"
rho_path = "/home/bugger/Documents/presentaties/Espresso/november2020/000029.00001.006.nrrd"
flavio_rx = "/home/bugger/Documents/presentaties/Espresso/november2020/M10_rx.npy"
flavio_tx = "/home/bugger/Documents/presentaties/Espresso/november2020/M10_tx.npy"
flavio_mask = "/home/bugger/Documents/presentaties/Espresso/november2020/M10_mask.npy"

"""
Load/prep data to show how it is used/constructed
"""

# Read mask
flavio_mask = np.load(flavio_mask)

# Read rho
rho_array, _ = nrrd.read(rho_path)
hplotc.SlidingPlot(np.rot90(np.moveaxis(rho_array, -1, 0), axes=(1,2))[:, ::-1])
sel_slice = 44
rho_array = rho_array[:, :, sel_slice]
rho_array = np.flipud(np.rot90(rho_array))
rho_array = harray.scale_minmax(rho_array)

# Get mask off rho
rho_mask = harray.get_treshold_label_mask(rho_array, treshold_value=0.1)
hplotc.ListPlot([rho_mask, rho_array])

# Read b1m registered
b1m_array = np.load(b1m_path)[0]
b1m_array = harray.scale_minmax(b1m_array, is_complex=True)

# Read b1p registered
b1p_array = np.load(b1p_path)[0]
b1p_array = harray.scale_minmax(b1p_array, is_complex=True)

# Shim B1p
mask_obj = hplotc.MaskCreator(b1p_array)
n_c, n_y, n_x = b1p_array.shape

shimming_obj = mb1.ShimmingProcedure(b1p_array, mask_obj.mask, relative_phase=True, str_objective='flip_angle')
x_opt, final_value = shimming_obj.find_optimum()
tx_amp = np.abs(x_opt)
tx_phase = np.angle(x_opt)

b1p_array_shim_not_scaled = apply_shim(b1p_array, amp_phase=(tx_amp, tx_phase))
b1p_array_shim = scale_signal_model(b1p_array_shim_not_scaled, flip_angle=np.pi/4)

biasfield = np.abs(b1p_array_shim * b1m_array).sum(axis=0)
biasfield_per_coil = b1p_array_shim * b1m_array
hplotc.ListPlot([b1p_array_shim_not_scaled], augm='np.abs', ax_off=True, cmap='viridis')
hplotc.ListPlot([b1p_array_shim], augm='np.abs', ax_off=True, cmap='viridis')
hplotc.ListPlot(b1m_array[None], augm='np.abs', ax_off=True, start_square_level=2, cmap='viridis')

hplotc.ListPlot([[rho_array]])
hplotc.ListPlot([[rho_array * biasfield, biasfield, rho_array * rho_mask]], augm='np.abs', ax_off=True)
hplotc.ListPlot([biasfield, b1p_array_shim, b1p_array_shim, b1p_array], augm='np.abs', cmap='viridis')
hplotc.ListPlot([biasfield_per_coil], augm='np.abs', cmap='viridis', ax_off=True, start_square_level=2,
                wspace=0, aspect_mode='auto', vmin=(0, 0.51))
hplotc.ListPlot([biasfield], augm='np.abs', cmap='viridis', ax_off=True, start_square_level=2,
                wspace=0, aspect_mode='auto', vmin=(0, 0.51))
hplotc.ListPlot([rho_array * biasfield_per_coil], augm='np.abs', cmap='gray', ax_off=True, start_square_level=2,
                wspace=0, aspect_mode='auto', vmin=(0, 0.2))
"""
Use this example in a model..
"""
# Here load the model etc...
import objective.inhomog_removal.executor_inhomog_removal as executor
import os
import helper.misc as hmisc
model_path_dir = '/home/bugger/Documents/model_run/inhom_removal_biasfield_2021'  # ==>
# model_path_dir = '/home/bugger/Documents/model_run/inhom_removal_undistorted_2021'  # ==>

# Overview of all the models available in model_path_dir
model_path_list = [os.path.join(model_path_dir, x) for x in os.listdir(model_path_dir)]
model_path_list = [x for x in model_path_list if 'result' not in x]

i_model_path = model_path_list[0]
model_name = os.path.basename(i_model_path)
print(i_model_path)

config_param = hmisc.convert_remote2local_dict(i_model_path, path_prefix='/media/bugger/MyBook/data/semireal')
# Otherwise squeeze will not work properly..
config_param['data']['batch_size'] = 1

decision_obj = executor.DecisionMaker(config_file=config_param, debug=False, load_model_only=True,
                                      inference=False, device='cpu')
modelrun_obj = decision_obj.decision_maker()
modelrun_obj.load_weights()

import torch
input_array = biasfield_per_coil * rho_array
im_y, im_x = input_array.shape[-2:]  # Get the last two indices
input_array = harray.to_stacked(input_array, cpx_type='cartesian', stack_ax=0)
input_array = input_array.T.reshape((im_x, im_y, -1)).T
input_tens = torch.from_numpy(input_array[np.newaxis]).float()
res = modelrun_obj.model_obj(input_tens)
hplotc.ListPlot([[biasfield, res.detach()[0][0] * rho_mask]], ax_off=True)

"""
Check EPT on Flavios data
"""
# Read flaio b1m
rx_array = np.load(flavio_rx)
rx_array = harray.scale_minmax(rx_array, is_complex=True)

# # Read flavio b1p
tx_array = np.load(flavio_tx)
tx_array = harray.scale_minmax(tx_array, is_complex=True)

# Test Helmholtz on flavio b1m and b1m registered
import helper.spacy as hspacy
import torch
res_flavio = hspacy.check_helmholtz(torch.as_tensor(tx_array[None, 0:1].real).float(), dx=0.01, wave_number=np.pi*3)
res_regist = hspacy.check_helmholtz(torch.as_tensor(b1p_array[None, 0:1].real).float(), dx=0.01, wave_number=np.pi*3)
hplotc.ListPlot([res_flavio, res_regist])