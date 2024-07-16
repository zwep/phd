import os
import shutil
from objective_configuration.reconstruction import DPRETRAINED, DMODEL, MODEL_NAMES

"""
We also need a 0p training...
We dont want to do this manually..

Actually.. doing it like this messes stuff up.. Because we are copying pre-trained weights.
This is OK for RADIAL and CIRCUS, but not for SCRATCH

This is now fixed since we copy 0pt models later
"""

list_of_pretrained_model = []
for i_model in MODEL_NAMES:
    model_path = os.path.join(DPRETRAINED, i_model)
    list_files = os.listdir(model_path)
    # Filter out all the non-.pt files and the model_0.pt file that I manually copied there
    filter_pt = sorted([x for x in list_files if x.endswith('pt') and x != 'model_0.pt'])
    # Select the one with the least iterations
    sel_pt_file = filter_pt[0]
    source_pt_file = os.path.join(model_path, sel_pt_file)
    source_config_file = os.path.join(model_path, 'config.yaml')
    for i_subsubdir in ['PRETR_ACQ', 'PRETR_SYNTH']:
        ddest = os.path.join(DMODEL, i_model + f"_{i_subsubdir}", '0p', 'pretrained')
        ddest_pt = os.path.join(ddest, sel_pt_file)
        ddest_config = os.path.join(ddest, 'config.yaml')
        if not os.path.isdir(ddest):
            print('Creating directory ', ddest)
            os.makedirs(ddest)
        #
        print(f'Copy {source_config_file} to {ddest_config}')
        shutil.copy(source_config_file, ddest_config)
        print(f'Copy {source_pt_file} to {ddest_pt}')
        shutil.copy(source_pt_file, ddest_pt)

    # Copy the 0 iteration model only if it is there
    # These are manually created... by starting a traiing seession and immediatly stopping it
    zero_pt_file = 'model_0.pt'
    source_pt_file = os.path.join(model_path, zero_pt_file)
    if os.path.isfile(source_pt_file):
        for i_subsubdir in ['SCRATCH_ACQ', 'SCRATCH_SYNTH']:
            ddest = os.path.join(DMODEL, i_model + f"_{i_subsubdir}", '0p', 'pretrained')
            ddest_pt = os.path.join(ddest, zero_pt_file)
            ddest_config = os.path.join(ddest, 'config.yaml')
            if not os.path.isdir(ddest):
                print('Creating directory ', ddest)
                os.makedirs(ddest)
            #
            print(f'Copy {source_config_file} to {ddest_config}')
            shutil.copy(source_config_file, ddest_config)
            print(f'Copy {source_pt_file} to {ddest_pt}')
            shutil.copy(source_pt_file, ddest_pt)


