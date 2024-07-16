"""
Here we update the base template
"""

import getpass
import os
import json
import sys
import helper.misc as hmisc

if os.path.isfile(__file__):
    dest_path = os.path.dirname(__file__)
else:
    if getpass.getuser() == 'bugger':
        dest_path = '/home/bugger/PycharmProjects/pytorch_in_mri/objective/unet_validation'
    else:
        dest_path = '/home/seb/code/pytorch_in_mri/objective/unet_validation'

print('dest path', dest_path)

basetemplate_dir = os.path.join(os.path.dirname(dest_path), 'default_configuration')
dest_dir = os.path.join(dest_path, "configuration")

base_template_name = "default_template.json"
model_template_name = "model_template.json"
target_name = "unet_validation.json"  # ==>

template = {"name": "unet_3D",
            "packed_keys": [],
            "dir": {"ddata":  "/data/seb/grand_challenge/data"},
            "data": {"num_workers": 0,  # Ehhh. yeah Maybe
                     "batch_perc": 0.0001,
                     "batch_size": 1,
                     "number_of_examples": None,
                     "input_shape": [132, 132, 116]},  # This can be changes according to the amount of files..
            "model": {"model_choice": "regular",

                      "n_epoch": 250,

                      },
            "optimizer": {"name": "Adam",
                          "config": {"lr": 0.001}},
            "callback": {"breakdown_limit": 20,
                         "memory_length": 15,
                         "memory_time": 20}
            }



"""
Base Template
"""

# Read base template
with open(os.path.join(basetemplate_dir, base_template_name), 'r') as f:
    base_template = json.loads(f.read())

"""
Model Template
"""
# Read model template
with open(os.path.join(dest_dir, model_template_name), 'r') as f:
    model_template = json.loads(f.read())

# Add the model configuration to the final output
model_choice = template['model']['model_choice'].lower()
model_key_name = 'config_' + model_choice
model_config = model_template.get(model_key_name, None)
if model_config is not None:
    template['model'].setdefault(model_key_name, model_config)

if model_choice == "regular":
    network_choice = model_config['network_choice']
    model_key_name = 'config_' + network_choice
    model_config = model_template.get(model_key_name, None)
    if model_config is not None:
        template['model'].setdefault(model_key_name, model_config)

# If model choice is gan.. we need to add the config of the generator and discriminator
if model_choice == "gan":
    generator_choice = model_config['generator_choice']
    generator_key_name = 'config_' + generator_choice
    generator_config = model_template.get(generator_key_name, None)
    if generator_choice is not None:
        template['model'].setdefault(generator_key_name, generator_config)

    discriminator_choice = model_config['discriminator_choice']
    discriminator_key_name = 'config_' + discriminator_choice
    discriminator_config = model_template.get(discriminator_key_name, None)
    if discriminator_config is not None:
        template['model'].setdefault(discriminator_key_name, discriminator_config)

# If model choice is gan.. we need to add the config of the two generators and two discriminator
# These are given by the keys generator_A and generator_B.. similar for discriminator
if model_choice == "cyclegan":
    # We are not able to process same architecture types with different config..
    generator_choice = model_config['generator_A_choice']
    generator_key_name = 'config_' + generator_choice
    generator_config = model_template.get(generator_key_name, None)
    if generator_choice is not None:
        template['model'].setdefault(generator_key_name, generator_config)

    generator_choice = model_config['generator_B_choice']
    generator_key_name = 'config_' + generator_choice
    generator_config = model_template.get(generator_key_name, None)
    if generator_choice is not None:
        template['model'].setdefault(generator_key_name, generator_config)

    discriminator_choice = model_config['discriminator_A_choice']
    discriminator_key_name = 'config_' + discriminator_choice
    discriminator_config = model_template.get(discriminator_key_name, None)
    if discriminator_config is not None:
        template['model'].setdefault(discriminator_key_name, discriminator_config)

    discriminator_choice = model_config['discriminator_B_choice']
    discriminator_key_name = 'config_' + discriminator_choice
    discriminator_config = model_template.get(discriminator_key_name, None)
    if discriminator_config is not None:
        template['model'].setdefault(discriminator_key_name, discriminator_config)

"""
Template
"""

import importlib
importlib.reload(hmisc)
hmisc.write_template(base_template, template, dest_dir, target_name, debug=True)
