import os
from pandas.io import clipboard
import re

"""

"""


def read_lines(line_list):
    z = []
    temp = []
    for i_line in line_list:
        if i_line.strip() == '':
            if len(temp):
                z.append(temp)
                temp = []
            continue
        else:
            temp.append(i_line)
    #
    return z


def get_key_range(start_stop, key_dict):
    # start_stop is a tuple
    if len(start_stop) == 1:
        start_stop = start_stop + start_stop
    s = ''
    for ii in range(start_stop[0], start_stop[1]+1):
        s += key_dict[str(ii)] + ", "
    else:
        s = s[:-2]
    #
    s = "\cite{" + s +"}"
    clipboard.copy(s)
    return s


# ddata = '/home/bugger/Documents/paper/14T/latex version/bib14T.bib'
# ddata_raw = '/home/bugger/Documents/paper/14T/latex version/reference_14T.txt'
ddata = '/home/bugger/Documents/paper/inhomogeneity removal/latex_version/reference_inhom.bib'
ddata_raw = '/home/bugger/Documents/paper/inhomogeneity removal/latex_version/reference_inhom.txt'


with open(ddata, 'r')   as f:
    A = f.readlines()


with open(ddata_raw, 'r') as f:
    B = f.readlines()

A_bibtex = read_lines(A)
B_raw = read_lines(B)

dict_bib_raw = {}
for i_ref, i_bib in zip(B_raw, A_bibtex):
    int_str = re.findall("\[([0-9]+)\]", i_ref[0])[0]
    i_bib_label = re.findall("\{([a-z0-9]+),", i_bib[0])[0]
    dict_bib_raw.update({int_str: i_bib_label})


z = get_key_range(key_dict=dict_bib_raw, start_stop=(27,))
for i in range(1, len(dict_bib_raw)+1):
    _ = get_key_range(key_dict=dict_bib_raw, start_stop=(i,))
    input(i)

z = get_key_range(key_dict=dict_bib_raw, start_stop=(41,))
get_key_range(key_dict=dict_bib_raw, start_stop=(28,))


#  Print out dir statistics... how far are we
# Gathering local file count
import os
dsync = '/media/bugger/MyBook/data/7T_scan/cardiac'

counter_sync = 0
filesize_sync = 0
for d, _, f in os.walk(dsync):
    if len(f):
        counter_sync += len(f)
        filesize_sync += sum([os.path.getsize(os.path.join(d, x)) for x in f])

import os
counter_sync = 9896
filesize_sync = 460927100819  # max 500 Gb
dd = '/smb/user/sharreve/BLD_RT_RESEARCH_DATA/backup'

counter = 0
filesize = 0
for d, _, f in os.walk(dd):
    if len(f):
        counter += len(f)
        filesize += sum([os.path.getsize(os.path.join(d, x)) for x in f])

print(f'Completed {counter} / {counter_sync}: {counter / counter_sync}')
print(f'Completed filesize: {filesize / filesize_sync}')

