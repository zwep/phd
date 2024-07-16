import os
import sys
import json
import re

def count_files(ddir):
    res_dict = {}
    for d, _, f in os.walk(ddir):
        if len(f):
            # print(d, len(f))
            res_dict.update({d: len(f)})
    return res_dict


if __name__ == "__main__":
    remote_count_str = '{"/data/seb/semireal/prostate_simulation_t1t2_rxtx/test/target_clean": 4, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/test/target_t1": 5, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/test/target_shimsettings": 8, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/test/target": 19, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/test/target_t2": 5, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/test/input": 19, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/train_old/target": 40, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/train_old/input": 40, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/validation/input": 1, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/validation/target_clean": 1, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/validation/target_shimsettings": 2, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/validation/target_t1": 1, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/validation/target": 1, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/validation/target_t2": 1, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/test_old/input": 60, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/test_old/target": 60, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/train/target_clean": 13, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/train/target_t1": 13, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/train/target_shimsettings": 24, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/train/target_t2": 13, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/train/input": 120, "/data/seb/semireal/prostate_simulation_t1t2_rxtx/train/target": 120, "/data/seb/semireal/prostate_simulation_rxtx/test_old/input": 3432, "/data/seb/semireal/prostate_simulation_rxtx/train/input": 6408, "/data/seb/semireal/prostate_simulation_rxtx/train/target": 8544, "/data/seb/semireal/prostate_simulation_rxtx/train/target_shimsettings": 24, "/data/seb/semireal/prostate_simulation_rxtx/train/target_clean": 14, "/data/seb/semireal/prostate_simulation_rxtx/train/mask": 534, "/data/seb/semireal/prostate_simulation_rxtx/train_old/input": 2136, "/data/seb/semireal/prostate_simulation_rxtx/validation/target_clean": 6, "/data/seb/semireal/prostate_simulation_rxtx/validation/input": 338, "/data/seb/semireal/prostate_simulation_rxtx/validation/target": 338, "/data/seb/semireal/prostate_simulation_rxtx/validation/mask": 286, "/data/seb/semireal/prostate_simulation_rxtx/validation/target_shimsettings": 4, "/data/seb/semireal/prostate_simulation_rxtx/test/target": 4238, "/data/seb/semireal/prostate_simulation_rxtx/test/mask": 286, "/data/seb/semireal/prostate_simulation_rxtx/test/input": 806, "/data/seb/semireal/prostate_simulation_rxtx/test/target_clean": 6, "/data/seb/semireal/prostate_simulation_rxtx/test/target_shimsettings": 8, "/data/seb/semireal/cardiac_simulation_rxtx/non_registered/b1minus/sel_filtered_aligned": 20, "/data/seb/semireal/cardiac_simulation_rxtx/non_registered/b1minus/leftovers": 79, "/data/seb/semireal/cardiac_simulation_rxtx/non_registered/b1minus/sel_filtered": 20, "/data/seb/semireal/cardiac_simulation_rxtx/non_registered/rho": 44, "/data/seb/semireal/cardiac_simulation_rxtx/non_registered/b1plus": 14, "/data/seb/semireal/cardiac_simulation_rxtx/registered": 1, "/data/seb/semireal/cardiac_simulation_rxtx/registered/test/target_clean": 14, "/data/seb/semireal/cardiac_simulation_rxtx/registered/test/target": 142, "/data/seb/semireal/cardiac_simulation_rxtx/registered/test/mask": 14, "/data/seb/semireal/cardiac_simulation_rxtx/registered/test/input": 56, "/data/seb/semireal/cardiac_simulation_rxtx/registered/train/mask": 30, "/data/seb/semireal/cardiac_simulation_rxtx/registered/train/input": 118, "/data/seb/semireal/cardiac_simulation_rxtx/registered/train/target": 287, "/data/seb/semireal/cardiac_simulation_rxtx/registered/train/target_clean": 30, "/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered/test/mask": 6, "/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered/test/target_clean": 6, "/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered/test/input": 6, "/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered/test/target": 6, "/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered/train/target_clean": 196, "/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered/train/mask": 196, "/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered/train/input": 196, "/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered/train/target": 196, "/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered/validation/target": 1, "/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered/validation/target_clean": 1, "/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered/validation/mask": 1, "/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered/validation/input": 1, "/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered/test/mask": 13, "/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered/test/input": 13, "/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered/test/target": 13, "/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered/test/target_clean": 13, "/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered/validation/input": 1, "/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered/validation/mask": 1, "/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered/validation/target_clean": 1, "/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered/validation/target": 1, "/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered/train/target_clean": 166, "/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered/train/input": 166, "/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered/train/mask": 166, "/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered/train/target": 166, "/data/seb/semireal/prostate_simulation_h5/train/target_clean": 11, "/data/seb/semireal/prostate_simulation_h5/train/mask": 11, "/data/seb/semireal/prostate_simulation_h5/train/input_15": 176, "/data/seb/semireal/prostate_simulation_h5/train/target": 176, "/data/seb/semireal/prostate_simulation_h5/train/input": 176, "/data/seb/semireal/prostate_simulation_h5/test/input": 47, "/data/seb/semireal/prostate_simulation_h5/test/target_clean": 6, "/data/seb/semireal/prostate_simulation_h5/test/target": 47, "/data/seb/semireal/prostate_simulation_h5/test/mask": 6, "/data/seb/semireal/prostate_simulation_h5/test/input_15": 43, "/data/seb/semireal/prostate_simulation_h5/validation/mask": 1, "/data/seb/semireal/prostate_simulation_h5/validation/input": 1, "/data/seb/semireal/prostate_simulation_h5/validation/target": 1, "/data/seb/semireal/prostate_simulation_h5/validation/input_15": 1, "/data/seb/semireal/prostate_simulation_h5/validation/target_clean": 1}'
    remote_dict = json.loads(remote_count_str)

    ddir_book = "/media/bugger/MyBook/data/semireal/semireal"
    local_dict = count_files(ddir_book)

    replace_str_remote = "/data/seb/semireal/"
    orig_keys = list(remote_dict.keys())
    for k in orig_keys:
        new_key = re.sub(replace_str_remote, "", k)
        remote_dict[new_key] = remote_dict[k]
        remote_dict.pop(k, None)

    replace_str_local = "/media/bugger/MyBook/data/semireal/semireal/"
    for k in list(local_dict.keys()):
        new_key = re.sub(replace_str_local, "", k)
        local_dict[new_key] = local_dict[k]
        del local_dict[k]

    all_keys = set(local_dict.keys()).union(set(remote_dict.keys()))
    space_1 = 60
    space_2 = 7
    for i_key in all_keys:
        true_value = remote_dict.get(i_key, 0)
        cur_value = local_dict.get(i_key, 0)
        if true_value != cur_value:
            print(i_key, ' '* (space_1 - len(i_key)), true_value, ' ' * (space_2 - len(str(true_value))), cur_value)