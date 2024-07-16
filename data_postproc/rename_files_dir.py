
import shutil
import os
import re

# orig_dir = '/home/bugger/Documents/data/semireal/prostate_simulation_t1t2_rxtx/validation/target_clean/'
orig_dir = '/data/cmr7t3t/mms1/all_phases_mid/Vendor_A/Results/ACDC_Synth_Model_21_02_22'
list_files = os.listdir(orig_dir)
orig_text = '_0000'
sub_text = ''


for i_file in list_files:
    old_name = i_file
    new_name = re.sub(orig_text, sub_text, old_name)
    shutil.move(os.path.join(orig_dir, old_name),
                os.path.join(orig_dir, new_name))

