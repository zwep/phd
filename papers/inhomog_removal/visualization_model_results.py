import matplotlib.pyplot as plt

import helper.misc as hmisc
import os
import numpy as np
import helper.array_transf as harray
import helper.plot_class as hplotc
import helper.plot_fun as hplotf
from loguru import logger
from objective_configuration.inhomog_removal import get_path_dict
import argparse
import sys


def get_metric_from_glcm(list_of_model_result, glcm_dict, file_name, sel_slice):
    """
    The homogeneity measure is stored as a long list over all slices and files
    The selected slice information is stored in a list of lists. [[slice1, slice2, ... ], [slice1, slice2, ...], ...]
                                                                  | --- file 1 ---     |  | --- file 2 ---    | | ...|
    :param list_of_model_result:
    :param glcm_dict:
    :param file_name:
    :param sel_slice:
    :return:
    """
    if isinstance(list_of_model_result, str):
        list_of_model_result = [list_of_model_result]
    # The model result list determiens the order of the homog/energy list
    homog_energy_list = []
    for i_model in list_of_model_result:
        temp_file_list = glcm_dict[i_model]['file_list']
        temp_slice_list = glcm_dict[i_model]['slice_list']
        sel_file_index = hmisc.find_index_file(temp_file_list, file_name)
        sel_slice_list = temp_slice_list[sel_file_index]
        index_offset = sum([len(x) for x in temp_slice_list[:sel_file_index]])
        sel_slice_index = sel_slice_list.index(sel_slice)
        sel_index = index_offset + sel_slice_index
        homog_value = glcm_dict[i_model]['homogeneity'][sel_index]
        energy_value = glcm_dict[i_model]['energy'][sel_index]
        homog_energy_list.append((homog_value, energy_value))
    return homog_energy_list


"""
This testing has become the real thing...
"""

parser = argparse.ArgumentParser()
parser.add_argument('--inhomonet', default=False, action='store_true')
parser.add_argument('-dataset', type=str, default='all',
                    help='Provide the name of the dataset on which we want to evaluate: '
                         'synthetic, 3T, patient, volunteer')
parser.add_argument('--debug', default=False, action='store_true')

p_args = parser.parse_args()
# inhomonet_bool = False
# dataset = 'patient_3T'
# debug_bool = False
inhomonet_bool = p_args.inhomonet
dataset = p_args.dataset
debug_bool = p_args.debug

_, dataset_list = get_path_dict('')

if dataset == 'all':
    sel_dataset_list = dataset_list
else:
    if dataset in dataset_list:
        sel_dataset_list = [dataset]
    else:
        logger.debug(f'Unknown dataset selected: {dataset}')
        logger.debug(f'Possible datasets: {dataset_list}')
        sys.exit()


fontsize = 16

# Remote destination directory
dest_dir_main = '/home/sharreve/local_scratch/paper/inhomog/model_results'
if inhomonet_bool:
    dest_dir_main = dest_dir_main + '_inhomonet'


dest_dir_volunteer = os.path.join(dest_dir_main, 'volunteer')
dest_dir_test_split = os.path.join(dest_dir_main, 'test_split')
dest_dir_patient_7T = os.path.join(dest_dir_main, 'patient_7T')
dest_dir_patient_3T = os.path.join(dest_dir_main, 'patient_3T')

hmisc.check_and_create_dir([dest_dir_volunteer, dest_dir_test_split, dest_dir_patient_3T, dest_dir_patient_7T])

base_model_dir = '/home/sharreve/local_scratch/model_run/selected_inhomog_removal_models'
inhomonet_config_path = os.path.join(base_model_dir, 'inhomonet')
biasf_single_config_path = os.path.join(base_model_dir, 'single_biasfield')
homog_single_config_path = os.path.join(base_model_dir, 'single_homogeneous')
biasf_multi_config_path = os.path.join(base_model_dir, 'multi_biasfield')
homog_multi_config_path = os.path.join(base_model_dir, 'multi_homogeneous')
both_single_config_path = os.path.join(base_model_dir, 'single_both')
n4_config_path = os.path.join(base_model_dir, 'n4')

# patient_7T_n4 = os.path.join(n4_config_path, '/home/sharreve/local_scratch/mri_data/daan_reesink/image_n4itk/pred'
# patient_3T_n4 = '/home/sharreve/local_scratch/mri_data/prostate_weighting_h5/test/target_corrected_N4'
# volunteer_n4 = '/home/sharreve/local_scratch/mri_data/volunteer_data/t2w_n4itk/pred'
# test_n4 = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/corrected_N4_new'

if inhomonet_bool:
    mandatory_text_box_names = ['Uncorrected', 'Reference:\nN4(ITK) algorithm', 'Inhomonet']
    pred_config_all = [inhomonet_config_path]
    pred_config_single_channel = [inhomonet_config_path]
    model_text_box_names = []
else:
    # Combine the possible interesting configs
    mandatory_text_box_names = ['Uncorrected', 'Reference:\nN4(ITK) algorithm']
    pred_config_all = [biasf_single_config_path, homog_single_config_path, biasf_multi_config_path, homog_multi_config_path]
    pred_config_single_channel = [biasf_single_config_path, homog_single_config_path]
    # Text boxes
    model_text_box_names = ['Corrected:\nSingle channel t-Biasfield', 'Corrected:\nSingle channel t-Image ',
                            'Corrected:\nMulti channel t-Biasfield ', 'Corrected:\nMulti channel t-Image ']

# Design the text boxes...
optional_text_box_names = ['Target']

#
dict_volunteer = {'config_list': pred_config_all,
                  'dataset': 'volunteer_corrected',
                  'dataset_config_name': 'volunteer',
                  'ddest': dest_dir_volunteer,
                  'ext': '.nii.gz',
                  'file_list': ["v9_03032021_1647583_11_3_t2wV4.nii.gz",
                                "v9_11022021_1643158_8_3_t2wV4.nii.gz",
                                "v9_10022021_1725072_12_3_t2wV4.nii.gz",
                                "v9_18012021_0939588_10_3_t2wV4.nii.gz"]}

dict_patient_7T = {'config_list': pred_config_single_channel,
                   'dataset': 'patient_corrected',
                   'dataset_config_name': 'patient',
                   'ddest': dest_dir_patient_7T,
                   'ext': '.nii.gz',
                   'file_list': ["7TMRI002.nii.gz",
                                 "7TMRI005.nii.gz",
                                 "7TMRI016.nii.gz",
                                 "7TMRI020.nii.gz"]}

dict_patient_3T = {'config_list': pred_config_single_channel,
                   'dataset': 'patient_corrected_3T',
                   'dataset_config_name': 'patient_3T',
                   'ddest': dest_dir_patient_3T,
                   'ext': '.nii.gz',
                   # 'ext_n4': '.h5',
                   'file_list': ['8_MR.nii.gz',
                                 '19_MR.nii.gz',
                                 '41_MR.nii.gz',
                                 '45_MR.nii.gz']}

dict_test = {'config_list': pred_config_all,
             'dataset': 'target_corrected',
             'dataset_config_name': 'synthetic',
             'ddest': dest_dir_test_split,
             'ext': '.nii.gz',
             'file_list': ["M20_to_51_MR_20200928_0003_transversal.nii.gz",
                           "M23_to_48_MR_20210127_0002_transversal.nii.gz",
                           "M20_to_8_MR_20210401_0003_transversal.nii.gz",
                           "M20_to_9_MR_20210324_0009_transversal.nii.gz"]}

all_config_dict = [dict_test, dict_volunteer, dict_patient_3T, dict_patient_7T]
selected_config_dict = [x for x in all_config_dict if x['dataset_config_name'] in sel_dataset_list]
# I think it is best to first collect all the GLCM dictionaries..
# Can we do that here..?
# Well first test if the current changes are OK

for tempdict in selected_config_dict:
    # List of paths TO model results
    list_of_model_result = tempdict['config_list']
    dataset_midfix = tempdict['dataset']
    config_dict, _ = get_path_dict('')
    ddest = tempdict['ddest']
    dir_n4_results = os.path.join(n4_config_path, dataset_midfix, 'pred')
    file_ext_model_result = tempdict['ext']
    # file_list = os.listdir(dir_n4_results)
    file_list = tempdict['file_list']
    print('Using dataset ', dataset_midfix)
    print('Visualizating the following files ')
    for i_file in file_list:
        print(f'\t {i_file}')
    print('For these config-files ')
    for i_model in list_of_model_result:
        print(f'\t {i_model}')
    print()
    # Set the input, mask and target directories. These are the same for all config-files
    # I think changing these will benefit the overal visualization...
    # input_dir = os.path.join(list_of_model_result[0], dataset_midfix, 'input')
    input_dir = config_dict[tempdict['dataset_config_name']]['dimage']
    # mask_dir = config_dict[tempdict['dataset_config_name']]['dmask']
    mask_dir = os.path.join(list_of_model_result[0], dataset_midfix, 'mask')
    target_dir = config_dict[tempdict['dataset_config_name']]['dtarget']
    # Check if there are any target files that need to be loaded
    target_file_present = True if target_dir is not None else False
#
    """    Define the text boxes displayed above the images    """
#
    if 'patient' in dataset_midfix:
        # Take only the names of the single channel models (those are the first two)
        text_box_names = mandatory_text_box_names + model_text_box_names[:2]
    else:
        text_box_names = mandatory_text_box_names + model_text_box_names
#
    if target_file_present:
        text_box_names += optional_text_box_names
#
    """ Retrieve homogeneity dictionaryies """
#
    # Using this for now..
    # homogeneity_energy_list = [(0, 0)]
    # Get the homogeneity and energy list of the chosen image..
    # Collect the homogeneity and energy...
    import papers.inhomog_removal.helper_inhomog as helper_inhomog
    from objective_configuration.inhomog_removal import get_path_dict
    #
    pred_glcm_dict = {}
    input_glcm_dict = {}
    target_glcm_dict = {}
    # This is fine.. I guess.. but it leaves out the N4 predictions..
    # This does include the potential Inhomonet results
    for i_model in list_of_model_result:
        logger.info(f'\t With dataset {i_model} ')
        path_to_pred = os.path.join(i_model, dataset_midfix)
        logger.info(f'\t Path to prediction {path_to_pred} ')
        pred_glcm_dict = helper_inhomog.collect_glcm(path_to_pred, file_name='pred_change_glcm',
                                                     temp_dict=pred_glcm_dict, key=i_model)
        input_glcm_dict = helper_inhomog.collect_glcm(path_to_pred, file_name='input_change_glcm',
                                                      temp_dict=input_glcm_dict, key=i_model)
        if target_file_present:
            target_glcm_dict = helper_inhomog.collect_glcm(path_to_pred, file_name='target_change_glcm',
                                                           temp_dict=target_glcm_dict, key=i_model)
    # Also retrieve the N4 GLCM values
    n4_glcm_dict = {}
    path_n4_glcm = os.path.join(n4_config_path, dataset_midfix)
    n4_glcm_dict = helper_inhomog.collect_glcm(path_n4_glcm, file_name='pred_change_glcm',
                                               temp_dict=n4_glcm_dict, key=f'n4_{dataset_midfix}')
#
    # Now we are going to process each file and visualize that..
    for i_file in file_list:
        print(f'\t Processing file {i_file}')
        i_file_no_ext = hmisc.get_base_name(i_file)
        # Load the N4-corrected file
        ext_n4 = tempdict.get('ext_n4', tempdict['ext'])
        n4_file_path = os.path.join(dir_n4_results, i_file_no_ext + ext_n4)
        result_n4 = hmisc.load_array(n4_file_path)
        if ext_n4.endswith('.nii.gz'):
            result_n4 = result_n4.T[:, ::-1, ::-1]
#
        """         Load the corresponding input-/target-/mask-files        """
        input_file_path = os.path.join(input_dir, i_file_no_ext + file_ext_model_result)
        mask_file_path = os.path.join(mask_dir, i_file_no_ext + file_ext_model_result)
        input_array = hmisc.load_array(input_file_path)
        mask_array = hmisc.load_array(mask_file_path)
        # If there is a target file, load it
        if target_file_present:
            target_file_path = os.path.join(target_dir, i_file_no_ext + file_ext_model_result)
            target_array = hmisc.load_array(target_file_path)
            if 'nii' in file_ext_model_result:
                target_array = target_array.T[:, ::-1, ::-1]
        if 'nii' in file_ext_model_result:
            input_array = input_array.T[:, ::-1, ::-1]
        if 'nii' in file_ext_model_result:
            mask_array = mask_array.T[:, ::-1, ::-1]
#
        """         Extract the relevant Homog/Energy    """
        n_slice = result_n4.shape[0]
        # I think this should solve some problems..
        if n_slice > 1:
            sel_slice = n_slice//2 + 1
        else:
            sel_slice = 0
#
        # Indeed, this should not matter.. since all the models SHOULD have the same input GLCM metrics
        input_homog_energy = get_metric_from_glcm(list_of_model_result[0], glcm_dict=input_glcm_dict, file_name=i_file, sel_slice=sel_slice)
        n4_homog_energy = get_metric_from_glcm(f'n4_{dataset_midfix}', glcm_dict=n4_glcm_dict, file_name=i_file, sel_slice=sel_slice)
        pred_homog_energy = get_metric_from_glcm(list_of_model_result, glcm_dict=pred_glcm_dict, file_name=i_file, sel_slice=sel_slice)
#
        homogeneity_energy_list = []
        homogeneity_energy_list.extend(input_homog_energy)
        homogeneity_energy_list.extend(n4_homog_energy)
        homogeneity_energy_list.extend(pred_homog_energy)
#
        if target_file_present:
            # Similar here, all models should have the same target GLCM metrics. Taking the first one should not matter
            target_homog_energy = get_metric_from_glcm(list_of_model_result[0], glcm_dict=target_glcm_dict, file_name=i_file, sel_slice=sel_slice)
            homogeneity_energy_list.extend(target_homog_energy)
#
        """         Load all the predicted images        """
#
        model_result_list = []
        for i_pred_dir in list_of_model_result:
            pred_file_path = os.path.join(i_pred_dir, dataset_midfix, 'pred', i_file_no_ext + file_ext_model_result)
            loaded_array = hmisc.load_array(pred_file_path)
            if 'nii' in file_ext_model_result:
                loaded_array = loaded_array.T[:, ::-1, ::-1]
            model_result_list.append(loaded_array)
        print(f'\t Loaded all the data')
#
        # sel_slice = 0
        result_n4_sel = harray.scale_minmax(result_n4[sel_slice])
        input_array_sel = harray.scale_minmax(input_array[sel_slice])
        mask_array_sel = harray.scale_minmax(mask_array[sel_slice])
        if target_file_present:
            target_array_sel = target_array[sel_slice] * mask_array_sel
            target_array_sel = harray.scale_minmax(target_array_sel)
        model_result_list_sel = [harray.scale_minmax(x[sel_slice]) for x in model_result_list]
#
        """         Select appropriate patches...        """
#
        # These patches are used for correcting the pixel intensity
        if dataset_midfix == 'volunteer_corrected':
            # Needed for volunteer 7T data
            patch_size = 10 * 10
            dataset_name = 'volunteer'
        elif dataset_midfix == 'patient_corrected_3T':
            # Needed for patient 3T
            patch_size = 7 * 10
            dataset_name = 'patient_3T'
        elif dataset_midfix == 'patient_corrected':
            patch_size = 16 * 10
            dataset_name = 'patient'
        elif dataset_midfix == 'target_corrected':
            patch_size = 7 * 10
            dataset_name = 'test'
        else:
            dataset_name = ''
            patch_size = 10 * 10
            print('Unknown dataset name. Received: ', dataset_midfix)
#
        """         Equalize the images...        """
#
        if target_file_present:
            plot_array = np.array([input_array_sel] + [result_n4_sel] + model_result_list_sel + [target_array_sel])
        else:
            plot_array = np.array([input_array_sel] + [result_n4_sel] + model_result_list_sel)
#
        plot_array_orig = np.copy(plot_array)
        # # # # # #
        # Debug the Image Equalized
        # # # # # #
        if debug_bool:
            #
            n_images = len(plot_array)
            fig, ax = plt.subplots(2, figsize=(30, 10))
            nx, ny = plot_array[0].shape  # Assuming that all are the same shape
            x_line = [x[nx // 2, :] for x in plot_array]
            y_line = [x[:, ny // 2] for x in plot_array]
            print([x.shape for x in x_line])
            for ii, i_line in enumerate(x_line):
                ax[0].plot(i_line, label=text_box_names[ii])
            for ii, i_line in enumerate(y_line):
                ax[1].plot(i_line, label=text_box_names[ii])
            ax[0].legend()
            ax[1].legend()
            fig.savefig(os.path.join(ddest, i_file_no_ext + f"_{sel_slice}_lineplot_before" + '.png'),
                        bbox_inches='tight', pad_inches=0.0, dpi=300)

        # Perform mean scaling
        temp_crop_coords = harray.get_crop_coords_center(list(input_array_sel.shape), width=patch_size)
        mean_ref = np.mean(input_array_sel[temp_crop_coords[0]:temp_crop_coords[1], temp_crop_coords[2]:temp_crop_coords[3]])
        mean_scaled_plot_array = []
        for i_img in plot_array:
            mean_img = np.mean(i_img[temp_crop_coords[0]:temp_crop_coords[1], temp_crop_coords[2]:temp_crop_coords[3]])
            i_img = i_img * mean_ref / mean_img
            mean_scaled_plot_array.append(i_img)

        if debug_bool:
            # Plot the line graphs after we have before mean scaling
            fig, ax = plt.subplots(2, figsize=(30, 10))
            nx, ny = mean_scaled_plot_array[0].shape  # Assuming that all are the same shape
            x_line = [x[nx // 2, :] for x in mean_scaled_plot_array]
            y_line = [x[:, ny // 2] for x in mean_scaled_plot_array]
            print([x.shape for x in x_line])
            for ii, i_line in enumerate(x_line):
                ax[0].plot(i_line, label=text_box_names[ii])
            for ii, i_line in enumerate(y_line):
                ax[1].plot(i_line, label=text_box_names[ii])
            ax[0].legend()
            ax[1].legend()
            fig.savefig(os.path.join(ddest, i_file_no_ext + f"_{sel_slice}_lineplot_mean_scaled" + '.png'),
                        bbox_inches='tight', pad_inches=0.0, dpi=300)

        """         Create the array to be plotted      """
        #
        # This worked really well. So we re-define the plot array here
        # plot_array = mean_scaled_plot_array
        # Here we crop so that we have as little black bands as possible
        _, crop_coords = harray.get_center_transformation_coords(mask_array_sel)
        # plot_array = np.array([harray.apply_crop_axis(x, crop_coords=crop_coords, axis=0) for x in mean_scaled_plot_array])
        plot_array = np.array([harray.apply_crop_axis(x, crop_coords=crop_coords, axis=0) for x in plot_array])
        mask_array_sel = harray.apply_crop_axis(mask_array_sel, crop_coords=crop_coords, axis=0)

        print('Shape of plot array', [x.shape for x in plot_array])
        print('Shape of mask array', mask_array_sel.shape)

        plot_array = np.array([x * mask_array_sel for x in plot_array])

        vmax_list = [(0, harray.get_proper_scaled_v2(x, (patch_size, patch_size), patch_size//2)) for x in plot_array]
        print('Calculated vmax ', vmax_list)

        """         Plot the plot array       """
        fig_obj = hplotc.ListPlot(plot_array[None], ax_off=True, hspace=0, wspace=0, vmin=[vmax_list], figsize=(30, 10))
        fig = fig_obj.figure
        height_offset = 0.05

        for i, i_text_box_name in enumerate(text_box_names):
            hplotf.add_text_box(fig, i, i_text_box_name, height_offset=0, position='top', fontsize=fontsize)

        # Now add the bottom text boxes...
        for ii, (i_hom, i_energ) in enumerate(homogeneity_energy_list):
            i_hom = "%0.2f" % i_hom
            i_energ = "%0.2f" % i_energ
            hplotf.add_text_box(fig, ii, f'H:{i_hom}        E:{i_energ}', height_offset=0, position='bottom', fontsize=fontsize)

        fig.savefig(os.path.join(ddest, i_file_no_ext + f"_{sel_slice}" + '.png'),
                    bbox_inches='tight', pad_inches=0.0, dpi=300)
        hplotc.close_all()

        # This was needed for debugging purposes
        # dest_npy = os.path.join(ddest, i_file_no_ext + f"_{sel_slice}" + '.npy')
        # dest_textbox = os.path.join(ddest, i_file_no_ext + f"_{sel_slice}_text" + '.txt')
        # dest_hom_energy = os.path.join(ddest, i_file_no_ext + f"_{sel_slice}_metric" + '.txt')
        #
        # np.save(dest_npy, plot_array)
        #
        # with open(dest_textbox, 'w') as f:
        #     f.write(','.join(text_box_names))
        #
        # with open(dest_hom_energy, 'w') as f:
        #     f.write(','.join([str(x) for x in homogeneity_energy_list]))

        """
        Lets do the same for cropped images
        """

        zoom_size = 100
        if dataset_midfix == 'patient_corrected':
            zoom_size = 150
        if dataset_midfix == 'target_corrected':
            zoom_size = 100

        n_midy = model_result_list_sel[0].shape[0] // 2
        n_midx = model_result_list_sel[0].shape[1] // 2

        # And now crop so that we have a clear picture of the prostate
        crop_coords = (n_midx, n_midx, n_midy, n_midy)
        plot_array_crop = np.array([harray.apply_crop(x, crop_coords=crop_coords, marge=zoom_size) for x in plot_array_orig])
        # Re-use the old vmax-list
        fig_obj = hplotc.ListPlot(plot_array_crop[None], ax_off=True, hspace=0, wspace=0, vmin=[vmax_list], figsize=(30, 10))
        fig = fig_obj.figure
        height_offset = 0.05

        for i, i_text_box_name in enumerate(text_box_names):
            hplotf.add_text_box(fig, i, i_text_box_name, height_offset=0, position='top', fontsize=fontsize)

        plt.tight_layout()
        fig.savefig(os.path.join(ddest, i_file_no_ext + f"_{sel_slice}_crop" + '.png'),
                    bbox_inches='tight', pad_inches=0.0, dpi=300)
        hplotc.close_all()

        # This was needed for debugging purposes
        # dest_npy = os.path.join(ddest, i_file_no_ext + f"_{sel_slice}_crop" + '.npy')
        # dest_textbox = os.path.join(ddest, i_file_no_ext + f"_{sel_slice}_crop_text" + '.txt')
        #
        # np.save(dest_npy, plot_array_crop)
        #
        # with open(dest_textbox, 'w') as f:
        #     f.write(','.join(text_box_names))
        #
