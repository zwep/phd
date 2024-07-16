import objective.inhomog_removal.executor_inhomog_removal as executor
import objective.inhomog_removal.postproc_inhomog_removal as postproc_inhomog
import os


"""
Load the model
"""

biasfield_model_path = '/local_scratch/sharreve/model_run/inhomog_single/config_00'

"""
GET ZEH DATAA
"""


for data_type in ['validation', 'test', 'train']:
    data_dir = '/local_scratch/sharreve/mri_data/prostate_weighting_h5'
    target_dir = os.path.join(data_dir, data_type, 'target')
    mask_dir = os.path.join(data_dir, data_type, 'mask')
    file_list = os.listdir(target_dir)
    storage_dir = os.path.join(data_dir, data_type, 'target_corrected')

    postproc_obj = postproc_inhomog.PostProcInhomogRemoval(image_dir=target_dir,
                                                           mask_dir=mask_dir,
                                                           target_dir=storage_dir,
                                                           config_path=biasfield_model_path,
                                                           executor_module=executor, config_name='config_param.json',
                                                           stride=64, patch_shape=(256, 256),
                                                           storage_extension='h5',
                                                           mask_suffix='_target')
    postproc_obj.run()