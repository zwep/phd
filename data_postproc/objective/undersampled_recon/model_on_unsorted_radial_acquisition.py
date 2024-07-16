import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch
import helper.plot_class as hplotc
import helper.misc as hmisc
import objective.undersampled_recon.executor_undersampled_recon as executor
import objective.undersampled_recon.recall_undersampled_recon as recall
import helper.array_transf as harray
import reconstruction.ReadCpx as read_cpx
import reconstruction.ReadRec as read_rec


"""
We had some unsorted radial data (untriggered etc.)

Now we want to apply a model to this..
"""

# ddata_file = '/media/bugger/MyBook/data/7T_data/sorted_untriggered_data/matthijs_100p_untriggered.npy'
ddata_file = '/media/bugger/MyBook/data/7T_data/sorted_untriggered_data/matthijs_20p_untriggered.npy'

# dconfig = '/home/bugger/Documents/model_run/undersampled_recon/resnet_24_sep'
dconfig = '/home/bugger/Documents/model_run/various_us_recon/resnet_radial_10'
recall_obj = recall.RecallUndersampled(dconfig, config_name='config_param.json')
recall_obj.mult_dict['config_00']['dir']['doutput'] = dconfig
modelrun_obj = recall_obj.get_model_object(recall_obj.mult_dict['config_00'])

cpx_cartesian = np.load(ddata_file)
hplotc.ListPlot(np.abs(cpx_cartesian[0,0]))
# Weird box to get to (256, 256) shape
img_to_predict = cpx_cartesian[:, -8:, 125:381, 125:381]

x_size, y_size = img_to_predict.shape[-2:]
x_range = np.linspace(-x_size//2, x_size//2, x_size)
y_range = np.linspace(-y_size // 2, y_size // 2, y_size)
X, Y = np.meshgrid(x_range, y_range)
mask_array = torch.from_numpy(np.sqrt(X ** 2 + Y ** 2) <= x_size//2)[None].float()

result_card = []
interm_result = []
counter = 0
for i_card in img_to_predict:
    print('Processing slice ', counter)
    counter += 1
    input_array = i_card[None]
    # input_array = input_array.sum(axis=1, keepdims=True)
    input_array = harray.scale_minmax(input_array, is_complex=True)
    n_y, n_x = input_array.shape[-2:]
    x_inputed = harray.to_stacked(input_array, cpx_type='cartesian', stack_ax=0)
    x_inputed = x_inputed.T.reshape((n_x, n_y, -1)).T
    A_tensor = torch.as_tensor(x_inputed[np.newaxis]).float() * mask_array[None]

    if modelrun_obj.trained_modelrun_obj is not None:
        with torch.no_grad():
            A_tensor = modelrun_obj.trained_modelrun_obj.model_obj(A_tensor)
        interm_result.append(A_tensor.detach().numpy())

    with torch.no_grad():
        if modelrun_obj.config_param['model']['model_choice'] == 'gan':
            output = modelrun_obj.generator(A_tensor)
        elif modelrun_obj.config_param['model']['model_choice'] == 'cyclegan':
            output = modelrun_obj.netG_A2B(A_tensor)
        else:
            output = modelrun_obj.model_obj(A_tensor)

    if output.shape[1] > 1:
        output_cpx = output.numpy()[0][0] + 1j * output.numpy()[0][1]
        # Here we take the ABS of the output..
        output_abs = np.abs(output_cpx)
    else:
        # Output is either abs or real.. any case.. it is fine..
        output_abs = output.numpy()[0][0]

    result_card.append(output_abs)

result_card_array = np.array(result_card)[:, :, ::-1]

hplotc.SlidingPlot(result_card_array, vmin=(0,1))

