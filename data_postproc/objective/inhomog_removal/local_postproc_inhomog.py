"""
Local evaluation
"""

import objective.inhomog_removal.executor_inhomog_removal as executor
import objective.inhomog_removal.postproc_inhomog_removal as postproc_inhomog
import helper.plot_class as hplotc
import os

# needed these for some misc coding
import helper.array_transf as harray
import matplotlib.pyplot as plt
import numpy as np
import helper.misc as hmisc
import pydicom


for ii_dataset in [1, 2, 3]:
    input_dir = f'/home/bugger/Documents/data/7T/liver/Images2Seb/Dataset{ii_dataset}'
    database_name = os.path.basename(input_dir)
    dest_dir = f'/home/bugger/Documents/data/7T/liver/Images2Seb/{database_name}_result'
    file_list = [x for x in os.listdir(input_dir) if x.startswith('IM')]

    base_model_dir = '/home/bugger/Documents/model_run'
    biasf_single_config_path = os.path.join(base_model_dir, 'inhomog_removal_biasfield/resnet_2022_04_28')
    homog_single_config_path = os.path.join(base_model_dir, 'inhomog_removal_single/resnet_2022_04_01')
    both_single_config_path_v2 = os.path.join(base_model_dir, 'inhomog_single_both/resnet_2022_06_01')
    both_single_config_path = os.path.join(base_model_dir, 'inhomog_single_both/resnet_2022_06')

    # These are the dicts for patient evaluation.
    homog_single_dict_volunteer = {"dconfig": homog_single_config_path, "dimage": input_dir, "dmask": None,
                         "ddest": dest_dir}
    # These are the dicts for patient evaluation.
    biasf_single_dict_volunteer = {"dconfig": biasf_single_config_path, "dimage": input_dir, "dmask": None,
                         "ddest": dest_dir}

    both_single_dict_volunteer = {"dconfig": both_single_config_path, "dimage": input_dir, "dmask": None,
                         "ddest": dest_dir}

    both_single_dict_volunteer_v2 = {"dconfig": both_single_config_path_v2, "dimage": input_dir, "dmask": None,
                         "ddest": dest_dir}

    dict_list = [homog_single_dict_volunteer, biasf_single_dict_volunteer, both_single_dict_volunteer, both_single_dict_volunteer_v2]

    model_counter = 0
    for temp_dict in dict_list:
        stride = 64
        target_dir = temp_dict.get('dtarget', None)
        postproc_obj = postproc_inhomog.PostProcInhomogRemoval(image_dir=temp_dict['dimage'],
                                                               mask_dir=temp_dict['dmask'],
                                                               dest_dir=temp_dict['ddest'],
                                                               target_dir=target_dir,
                                                               config_path=temp_dict['dconfig'],
                                                               executor_module=executor, config_name='config_param.json',
                                                               stride=stride, patch_shape=(256, 256),
                                                               storage_extension='nii')
        # We can change this...
        postproc_obj.experimental_postproc_both = 3
        postproc_obj.load_file(file_index=0)
        file_result = postproc_obj.run_loaded_file_patched()
        img_dir = homog_single_dict_volunteer['dimage']
        dicom_file_input = os.path.join(img_dir, postproc_obj.file_list[0])
        dicom_obj = pydicom.read_file(dicom_file_input)
        max_value = dicom_obj.pixel_array.max()
        corrected_result = np.array([x['corrected'] for x in file_result])
        corrected_result = harray.scale_minmax(corrected_result, axis=(-2, -1))
        corrected_result = np.array([(max_value * x).astype(np.int16) for x in corrected_result])
        dicom_obj.PixelData = corrected_result.tobytes()
        dicom_obj.save_as(os.path.join(postproc_obj.dest_dir, f"IM_0002_corrected_model_{model_counter}.dcm"))
        model_counter+=1
