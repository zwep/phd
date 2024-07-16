from data_prep.objective.reconstruction.pretr_acq.create_config_files import DIRECT_update_config_dict
import helper.misc as hmisc

dlocal = '/home/bugger/Documents/data/config_25p_mixed.yaml'
temp_yaml = hmisc.load_yaml(dlocal)

del temp_yaml['additional_models']

temp_yaml['training']['lr']