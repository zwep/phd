import os
import re
import shutil
from objective_configuration.reconstruction import DPRETRAINED, DMODEL, MODEL_NAMES, \
    PRETR_ACQ_APPENDIX, PRETR_SYNTH_ACQ_APPENDIX

"""
We also need to copy the pretrained synth models..

"""

re_iter_num = re.compile('model_([0-9]+).pt')

list_of_pretrained_model = []
for i_model in MODEL_NAMES:
    # Pretrained on ACQ data..
    model_path = os.path.join(DMODEL, i_model + PRETR_ACQ_APPENDIX, '100p', 'train_mixed')
    if os.path.isdir(model_path):
        list_files = os.listdir(model_path)
        # Filter out all the non-.pt files and the model_0.pt file that I manually copied there
        filter_pt = sorted([x for x in list_files if x.endswith('pt') and x != 'model_0.pt'], key=lambda x: int(re_iter_num.findall(x)[0]))
        # Select the one with the most iterations
        sel_pt_file = filter_pt[-1]
        source_pt_file = os.path.join(model_path, sel_pt_file)
        source_config_file = os.path.join(model_path, 'config.yaml')
        # Copy this to acq_synth 0p
        ddest = os.path.join(DMODEL, i_model + PRETR_SYNTH_ACQ_APPENDIX, '0p', 'pretrained')
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
    else:
        print(f'Not present: {model_path}')