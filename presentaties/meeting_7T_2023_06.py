import os
import numpy as np
import helper.plot_class as hplotc
import helper.misc as hmisc
from objective_helper.reconstruction import convert_to_sos

"""
We need some pictures

- Image of Transverse input
- GIF of input
- GIF of target (radial)
- GIF of target (cart)
- GIF of prediction (model 1, 2, 3, 4..?)
- bash command for convergence results

"""

dpresentation = '/home/bugger/Documents/presentaties/7TMeeting/presentatie_2023_06'

# Image of transverse input (cartesian)
# This one is needed for a high quality cover image
ddest_cartesian_img = os.path.join(dpresentation, 'cartesian_img.png')
groundtruth_dir = '/media/bugger/MyBook/data/7T_data/cartesian_cardiac_cine/h5_data/transverse'
file_list = os.listdir(groundtruth_dir)
sel_file = file_list[2]
groundtruth_file = os.path.join(groundtruth_dir, sel_file)

cpx_cartesian = hmisc.load_array(groundtruth_file)
cpx_cartesian = np.squeeze(cpx_cartesian)
fig_obj = hplotc.ListPlot(cpx_cartesian[15][100:480, 22:550], ax_off=True)
fig_obj.figure.savefig(ddest_cartesian_img, bbox_inches='tight', pad_inches=0.0)

# GIF of target (cartesian)
## We actually also have this one..
ddest_cartesian_gif = os.path.join(dpresentation, 'cartesian_img.gif')
hmisc.convert_image_to_gif(cpx_cartesian, output_path=ddest_cartesian_gif)

# GIF of target (radial)
## We also have this one too..

# GIF of pred (model)
from objective_configuration.reconstruction import DRESULT_INFERENCE, ANATOMY_LIST, PERCENTAGE_LIST, DRESULT_INFERENCE_png
import helper.array_transf as harray

dinference_old = '/local_scratch/sharreve/mri_data/cardiac_radial'
model_dir = os.listdir(DRESULT_INFERENCE)
# selected_model = model_dir[2]
for selected_model in model_dir:
    # selected_percentage = str(PERCENTAGE_LIST[1]) + 'p'
    for selected_percentage in PERCENTAGE_LIST:
        str_percentage = str(selected_percentage) + 'p'
        # selected_anatomy = ANATOMY_LIST[1]
        for selected_anatomy in ANATOMY_LIST:
            pred_path = os.path.join(DRESULT_INFERENCE, selected_model, str_percentage, 'train_mixed', selected_anatomy, 'undersampled')
            input_path = os.path.join(dinference_old, selected_anatomy + "_split", "test")
            pred_files = [x for x in os.listdir(pred_path) if x.endswith('h5')]
            # selected_file = pred_files[0]
            for selected_file in pred_files:
                file_name = hmisc.get_base_name(selected_file)
                sel_pred_file = os.path.join(pred_path, selected_file)
                sel_input_file = os.path.join(input_path, selected_file)
                print('Pred file', os.path.isfile(sel_pred_file))
                print('Input file', os.path.isfile(sel_input_file))
                ddir_png = os.path.join(DRESULT_INFERENCE_png, selected_model, str_percentage, selected_anatomy)
                if not os.path.isdir(ddir_png):
                    os.makedirs(ddir_png)
                #
                ddest_png = os.path.join(ddir_png, file_name + '.png')
                pred_array = hmisc.load_array(sel_pred_file, data_key='reconstruction')
                input_array = hmisc.load_array(sel_input_file, data_key='kspace')
                n_card = input_array.shape[0]
                sel_card = n_card // 2
                sos_input_array = convert_to_sos(input_array[sel_card])
                sel_pred = harray.scale_minmax(pred_array[sel_card])
                sel_input = harray.scale_minmax(sos_input_array)
                plot_array = [sel_pred, sel_input]
                patch_size = sel_input.shape[0] // 4
                vmax_list = [(0, harray.get_proper_scaled_v2(x, (patch_size, patch_size), patch_size // 2)) for x in plot_array]
                fig_obj = hplotc.ListPlot([sel_pred, sel_input], vmin=vmax_list)
                fig_obj.figure.savefig(ddest_png)