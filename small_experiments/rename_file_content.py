"""
I forgot to write 01 .. instead I used 1 in the file names..

This rewrites stuff

"""
#  Rename stuff......
import os
import re
import shutil
ddata = '/data/seb/paper/14T/plot_body_thomas_mask_rmse_sar/optim_shim_recalc_sar'
for d, _, f in os.walk(ddata):
    filter_f = [x for x in f if x.endswith('json') and re.findall('opt_shim_[0-9].json', x)]
    if len(filter_f):
        print(d)
        for i_file in filter_f:
            src_file = os.path.join(d, i_file)
            int_num = re.findall('opt_shim_([0-9]).json', i_file)[0]
            new_file = f'opt_shim_{str(int_num).zfill(2)}.json'
            tgt_file = os.path.join(d, new_file)
            # print(src_file, tgt_file)
            shutil.move(src_file, tgt_file)
