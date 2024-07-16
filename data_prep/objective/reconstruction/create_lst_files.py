from objective_configuration.reconstruction import DDATA, DDATA_SYNTH
import os
import numpy as np

"""
I want to automate this...

But first lets do some hacky- things..
"""



# The folders are organized as...

# dataset
#   train
#       input
#           file1
#           ...
#   test
#       input
#           file1
#           ...
#   validation
#       input
#           file1
#           ...
#
#
# So that is why the code below works (with dirname and basename etc)
# The .lst files are thus stored under dataset/train/train.lst (etc.)
for d, _, f in os.walk(DDATA):
    filter_f = [x for x in f if x.endswith('h5')]
    if len(filter_f):
        n_files = len(filter_f)
        for i_factor in np.linspace(0, 1, 5)[1:]:
            sel_files = int(n_files * i_factor)
            factor_str = str(int(i_factor * 100))
            print('========================= ')
            print(f'       {d}      {i_factor}')
            print('========================= ')
            ddest = os.path.dirname(d)
            i_type = os.path.basename(ddest)
            dest_lst_file = os.path.join(ddest, i_type + f'_{factor_str}.lst')
            with open(dest_lst_file, 'w') as f:
                f.writelines(line + '\n' for line in filter_f[:sel_files])


# Also do this for Synth
for d, _, f in os.walk(DDATA_SYNTH):
    filter_f = [x for x in f if x.endswith('h5')]
    if len(filter_f):
        n_files = len(filter_f)
        for i_factor in np.linspace(0, 1, 5)[1:]:
            sel_files = int(n_files * i_factor)
            factor_str = str(int(i_factor * 100))
            print('========================= ')
            print(f'       {d}      {i_factor}')
            print('========================= ')
            ddest = os.path.dirname(d)
            i_type = os.path.basename(ddest)
            dest_lst_file = os.path.join(ddest, i_type + f'_{factor_str}.lst')
            with open(dest_lst_file, 'w') as f:
                f.writelines(line + '\n' for line in filter_f[:sel_files])
