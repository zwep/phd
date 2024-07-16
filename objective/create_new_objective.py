
"""
Tired of creating everything with my hands... here is an automated script
"""

import re
import shutil
import os

# These are the only two that really needs to be changed to copy all the files
name_new_objective = 'reconstruction'
class_suffix = 'Reconstruction'


base_dir = '~/PycharmProjects/pytorch_in_mri/objective'
base_dir = os.path.expanduser(base_dir)

prep_dir = os.path.join('~/PycharmProjects/pytorch_in_mri/data_prep/objective', name_new_objective)
prep_dir = os.path.expanduser(prep_dir)
postproc_dir = os.path.join('~/PycharmProjects/pytorch_in_mri/data_postproc/objective', name_new_objective)
postproc_dir = os.path.expanduser(postproc_dir)

objective_dir = os.path.join(base_dir, name_new_objective)
objective_config_dir = os.path.join(objective_dir, 'configuration')

base_data_template_dir = os.path.join(base_dir, 'default_configuration/default_data_template.json')
target_data_template_dir = os.path.join(objective_config_dir, 'data_template.json')

base_model_template_dir = os.path.join(base_dir, 'default_configuration/default_model_template.json')
target_model_template_dir = os.path.join(objective_config_dir, 'model_template.json')

default_executor_dir = os.path.join(base_dir, 'default_configuration/default_executor.txt')
target_executor_dir = os.path.join(objective_dir, f'executor_{name_new_objective}.py')

default_main_dir = os.path.join(base_dir, 'default_configuration/default_main.txt')
target_main_dir = os.path.join(objective_dir, f'main_{name_new_objective}.py')

default_create_template_dir = os.path.join(base_dir, 'default_configuration/default_create_template.txt')
target_create_template_dir = os.path.join(objective_dir, f'create_template.py')

text_space = 35
temp_str = 'New prep/postproc dir'
if not os.path.isdir(postproc_dir):
    print(temp_str, ' ' * (text_space - len(temp_str)), postproc_dir)
    os.makedirs(postproc_dir)
else:
    print('No', temp_str.lower(), 'created')
if not os.path.isdir(prep_dir):
    print(temp_str, ' ' * (text_space - len(temp_str)), prep_dir)
    os.makedirs(prep_dir)
else:
    print('No', temp_str.lower(), 'created')

temp_str = 'New objective dir'
if not os.path.isdir(objective_dir):
    print(temp_str, ' ' * (text_space - len(temp_str)), objective_dir)
    os.makedirs(objective_dir)
else:
    print('No', temp_str.lower(), 'created')

temp_str = 'New objective configuration dir'
if not os.path.isdir(objective_config_dir):
    print(temp_str, ' ' * (text_space - len(temp_str)), objective_config_dir)
    os.makedirs(objective_config_dir)
else:
    print('No', temp_str.lower(), 'created')

temp_str = 'New data template'
if not os.path.isfile(target_data_template_dir):
    print(temp_str, ' ' * (text_space - len(temp_str)), target_data_template_dir)
    shutil.copy(base_data_template_dir, target_data_template_dir)
else:
    print('No', temp_str.lower(), 'created')

temp_str = 'New model template'
if not os.path.isfile(target_model_template_dir):
    print(temp_str, ' ' * (text_space - len(temp_str)), target_model_template_dir)
    shutil.copy(base_model_template_dir, target_model_template_dir)
else:
    print('No', temp_str.lower(), 'created')

# Read/alter/write the default_executor
temp_str = 'New executor file'
if not os.path.isfile(target_executor_dir):
    print(temp_str, ' ' * (text_space - len(temp_str)), target_executor_dir)

    with open(default_executor_dir, 'r') as f:
        default_exec_text = f.read()

    default_exec_text = re.sub(":class_suffix:", class_suffix, default_exec_text)

    with open(target_executor_dir, 'w') as f:
        f.write(default_exec_text)
else:
    print('No', temp_str.lower(), 'created')

# Read/alter/write the default_executor
temp_str = 'New main file'
if not os.path.isfile(target_main_dir):
    print(temp_str, ' ' * (text_space - len(temp_str)), target_main_dir)

    with open(default_main_dir, 'r') as f:
        default_main_text = f.read()

    default_main_text = re.sub(":name_objective:", name_new_objective, default_main_text)
    default_main_text = re.sub(":class_suffix:", class_suffix, default_main_text)

    with open(target_main_dir, 'w') as f:
        f.write(default_main_text)
else:
    print('No', temp_str.lower(), 'created')


# Read/alter/write the default_executor
temp_str = 'New create template file'
if not os.path.isfile(target_create_template_dir):
    print(temp_str, ' ' * (text_space - len(temp_str)), target_create_template_dir)

    with open(default_create_template_dir, 'r') as f:
        default_create_template_text = f.read()

    default_create_template_text = re.sub(":name_objective:", name_new_objective, default_create_template_text)

    with open(target_create_template_dir, 'w') as f:
        f.write(default_create_template_text)
else:
    print('No', temp_str.lower(), 'created')