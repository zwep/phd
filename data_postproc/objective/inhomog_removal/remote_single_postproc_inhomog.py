import objective.inhomog_removal.executor_inhomog_removal as executor
import objective.inhomog_removal.postproc_inhomog_removal as postproc_inhomog
import helper.plot_class as hplotc
import os

# needed these for some misc coding
import helper.array_transf as harray
import helper.misc as hmisc
import matplotlib.pyplot as plt
import numpy as np

"""
A selected dictionary..
Make a habbit of evaluating online
"""

selected_config = '/local_scratch/sharreve/model_run/inhomog_biasfield/config_00'
# selected_config = '/local_scratch/sharreve/model_run/inhomog_single_biasf/config_00'
input_dir_volunteer = '/home/sharreve/local_scratch/mri_data/volunteer_data/t2w'
mask_dir_volunteer = '/home/sharreve/local_scratch/mri_data/volunteer_data/body_mask'

input_dir_patient = '/home/sharreve/local_scratch/mri_data/daan_reesink/image'
mask_dir_patient = '/home/sharreve/local_scratch/mri_data/daan_reesink/mask'

input_dir_patient_3T = '/home/sharreve/local_scratch/mri_data/prostate_weighting_h5/test/target'
mask_dir_patient_3T = '/home/sharreve/local_scratch/mri_data/prostate_weighting_h5/test/mask'


input_dir_test = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/input'
input_dir_test_som = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/input_abs_sum'
mask_dir_test = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/mask_b1'
target_dir_test = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/target'


dict_volunteer = {"dconfig": selected_config, "dimage": input_dir_volunteer,
                              "dmask": mask_dir_volunteer,
                              "ddest": os.path.join(selected_config, "volunteer_corrected")}

dict_test = {"dconfig": selected_config, "dimage": input_dir_test,
                         "dmask": mask_dir_test,
                         "ddest": os.path.join(selected_config, "test_corrected")}

dict_patient_3T = {"dconfig": selected_config, "dimage": input_dir_patient_3T,
                              "dmask": mask_dir_patient_3T,
                              "ddest": os.path.join(selected_config, "patient_corrected_3T")}

dict_patient = {"dconfig": selected_config, "dimage": input_dir_patient,
                              "dmask": mask_dir_patient,
                              "ddest": os.path.join(selected_config, "patient_corrected")}


# for sel_dict in [dict_test, dict_patient_3T, dict_volunteer, dict_patient]:
for sel_dict in [dict_test, dict_volunteer]:
    ddest_model = sel_dict['ddest']
    dimage = sel_dict['dimage']
    base_name = os.path.basename(ddest_model)
    hmisc.create_datagen_dir(ddest_model, data_list=[], type_list=['input', 'biasfield', 'pred', 'mask', 'target'])
    mask_ext = '.nii.gz'
    mask_suffix = ''
    stride = 64
    if 'volunteer' in base_name:
        mask_ext = '.npy'
        mask_suffix = ''
    elif base_name.endswith('3T'):
        mask_ext = '.h5'
        mask_suffix = '_target'
    elif base_name.endswith('patient_corrected'):
        mask_ext = '.npy'
        stride = 128
    postproc_obj = postproc_inhomog.PostProcInhomogRemoval(image_dir=sel_dict['dimage'], mask_dir=sel_dict['dmask'],
                                                           config_path=sel_dict['dconfig'],
                                                           executor_module=executor, config_name='config_param.json',
                                                           stride=stride,
                                                           patch_shape=(256, 256),
                                                           storage_extension='nii',
                                                           mask_ext=mask_ext,
                                                           mask_suffix=mask_suffix,
                                                           dest_dir=ddest_model)
    postproc_obj.file_list = postproc_obj.file_list[0:1]
    postproc_obj.run()
    sel_slice = postproc_obj.n_slices//2
    slice_result = postproc_obj.run_slice_patched(sel_slice)
    postproc_slice = postproc_obj.postproc_loaded(slice_result, slice_index=sel_slice)
    corrected_result = postproc_slice['corrected']
    corrected_result = harray.treshold_percentile_both(corrected_result, q=98)
    equalize_obj = hplotc.ImageIntensityEqualizer(reference_image=postproc_obj.sum_of_absolute_img[sel_slice],
                                                  image_list=[corrected_result])
    corrected_result = equalize_obj.correct_image_list()
    corrected_result = np.array(corrected_result)
    hplotc.ListPlot([corrected_result, postproc_obj.sum_of_absolute_img[sel_slice]], title=ddest_model)

