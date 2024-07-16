from objective_configuration.reconstruction import DRESULT, DMODEL, TYPE_NAMES, MODEL_NAMES
import os
import shutil

"""
After copying all the pre-trained models... I found out that the configs are not properly set

This is because the default config in the pretrained-folder by DIRECT is not altered when copying to the 0p folder
since we simply copy the whole folder..

Here I correct for that.. Yay

I do this for each 'selected' (or trained) model variation
"""

for i_type in TYPE_NAMES:
    print()
    for i_model in MODEL_NAMES:
        model_type = f'{i_model}{i_type}'
        print(model_type)
        # I think this is not necessary..?
        some_config_file = os.path.join(DMODEL, model_type, 'config', 'config_25p_mixed.yaml')
        dest_dir = os.path.join(DMODEL, model_type, '0p', 'pretrained')
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)

        if os.path.isfile(some_config_file):
            dest_file = os.path.join(dest_dir, 'config.yaml')
            shutil.copy(some_config_file, dest_file)

        # Be sure that we also have this one..?
        some_config_file = os.path.join(DMODEL, model_type, 'config', 'config_25p_mixed.yaml')
        dest_dir = os.path.join(DMODEL, model_type, 'config')
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)

        if os.path.isfile(some_config_file):
            dest_file = os.path.join(dest_dir, 'config_0p_mixed.yaml')
            shutil.copy(some_config_file, dest_file)

        # Same thing for inferences
        some_config_file = os.path.join(DMODEL, model_type, 'config', 'inference_config_25p_mixed.yaml')
        if os.path.isfile(some_config_file):
            dest_file = os.path.join(DMODEL, model_type, 'config', 'inference_config_0p_mixed.yaml')
            shutil.copy(some_config_file, dest_file)
