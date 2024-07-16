import os
import numpy as np
import helper.misc as hmisc
from objective_configuration.fourteenT import DDATA_KT_POWER, DMASK_THOMAS
from objective_helper.fourteenT import convert_thomas_array_to_me
import scipy.io

"""
Thomas uses a different script than me to create his mask..
His masks are different over all files, since there is some randomness or so...
I will need to take the intersection and store that mask..

This mask will eventually be used to calculate metrics on for both my and his method

"""

mat_files = os.listdir(DDATA_KT_POWER)

if __name__ == "__main__":
    # Loop over the debug files to extract the masks..
    debug_design_files = [x for x in mat_files if x.startswith('debug')]
    mask_files = []
    for sel_file in debug_design_files:
        split_file_name = sel_file.split('_')
        coil_name = split_file_name[2]
        kt_number = hmisc.get_base_name(split_file_name[3])
        # Load the data...
        sel_mat_file = os.path.join(DDATA_KT_POWER, sel_file)
        mat_obj = scipy.io.loadmat(sel_mat_file)
        temp_mask = mat_obj['maps']['mask'][0][0]
        mask_files.append(temp_mask)

    # Plot all masks..
    mask_array = np.array(mask_files)
    # Transpose the array to have it in the correct order...
    mask_array = np.prod(mask_array, axis=0).T
    new_mask = convert_thomas_array_to_me(mask_array)
    np.save(DMASK_THOMAS, new_mask)
