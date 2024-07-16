
"""
Unfold everything using the correcting files that I created manually
"""

import os
import re
import data_prep.unfolding_data.ProcessVnumber as proc_vnumber


def get_v_number(path):
    scan_files = {}
    for d, sd, f in os.walk(path):
        regex_vnumber = re.findall('V9_[0-9]*', d)
        if regex_vnumber:
            v_number = regex_vnumber[0]
            scan_files.setdefault(v_number, [])

    return scan_files


if __name__ == "__main__":
    scan_dir = '/media/bugger/MyBook/data/7T_scan/cardiac'
    target_dir = '/media/bugger/MyBook/data/7T_data/unfolded_cardiac'

    vnumber_dict = get_v_number(scan_dir)
    # This is how we get all them v-numbers...
    unique_v_numbers = list(sorted(vnumber_dict.keys()))
    continue_index = unique_v_numbers.index('V9_16830')
    for v_number in unique_v_numbers[continue_index:]:
        print('Processing v number ', v_number)
        proc_obj = proc_vnumber.ProcessVnumber(v_number, scan_dir=scan_dir,
                                               target_dir=target_dir, debug=True,
                                               status=True, save_format='npy')
        proc_obj.run()
