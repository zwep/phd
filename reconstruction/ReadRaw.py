"""
Hoping that we can create a raw data reader....
"""
import numpy as np
import matplotlib.pyplot as plt
import os
raw_dir = '/media/bugger/WORK_USB/2021_12_01/ca_29045/'
raw_file_list = [os.path.join(raw_dir, x) for x in os.listdir(raw_dir) if x.endswith('raw')]
length_list = []
for raw_file in raw_file_list:
    # with open(raw_file, 'rb') as f_id:
    #     read_raw_file = f_id.read()
    with open(raw_file, 'rb') as f_id:
        read_raw_file = np.fromfile(f_id, count=-1, dtype=np.int16)
    n_length = len(read_raw_file)
    length_list.append(n_length)

for i_length in length_list:
    print(i_length, [i for i in range(0, 256, 1)[1:] if i_length % i == 0])
    # print("\t remainder ", i_length - 16 * int(i_length/16))

# Dus de gelezen data is niet een veelvoud van 16.. in geen geval...

#length_list
# Matlab code
# %% Read all hexadecimal labels
# source=>output... I dont understand that...
# [unparsed_labels, readsize] = fread (labfid,[16 Inf], 'uint32=>uint32');
# info.nLabels = size(unparsed_labels,2);

# When using count=-1 I get a value that isnt a multiple of 16
# So I floored that number so it is one of x16..
# That means that I leave out 4 bits..
with open(raw_file, 'rb') as f_id:
    read_raw_file = f_id.read()

len(read_raw_file) / 16

with open(raw_file, 'rb') as f_id:
    temp_file = np.fromfile(f_id, count=-1, dtype=np.uint32)
    n_length = len(temp_file)
    n_count = 16 * int(n_length / 16)

with open(raw_file, 'rb') as f_id:
    raw_fromfile = np.fromfile(f_id, count=n_count, dtype=np.uint32).reshape((-1, 16)).T


import binascii
with open(raw_file, 'rb') as f_id:
    hexdata = f_id.read().hex()

while True:
        hexdata = in_file.read(16).hex()     # I like to read 16 bytes in then new line it.
        if len(hexdata) == 0:                # breaks loop once no more binary data is read
            break

len(hexdata)
hexdata[0:100]
len(raw_fromfile)
raw_fromfile.dtype
#
raw_fromfile[0]
CardiacPhase = (raw_fromfile[7] & (2 ** 32-1)) >> 16

fig, ax = plt.subplots(2)
ax[0].plot(raw_fromfile[:, :2000].T)
ax[1].plot(raw_fromfile[:2000])

# # Dit gedeelte laat zien hoe je de data moet inlezen..
# info.labels.DataSize.vals         = unparsed_labels(1,:);
#
# info.labels.LeadingDummies.vals   = bitshift (bitand(unparsed_labels(2,:), (2^16-1)),  -0);
# info.labels.TrailingDummies.vals  = bitshift (bitand(unparsed_labels(2,:), (2^32-1)), -16);
#
"""
Verkeerd gelezen... had LAB moeten hebben
"""

import numpy as np
import matplotlib.pyplot as plt
import os
lab_file_name = '/media/bugger/WORK_USB/2021_12_01/ca_29045/ca_01122021_1021250_19_2_transverse_dyn_100p_radial_no_triggerV4.lab'


with open(lab_file_name, 'rb') as f_id:
    read_raw_file = np.fromfile(f_id, count=-1, dtype=np.uint32).reshape(-1, 16).T
# Gives a size of 16 x 5336... too little...

with open(lab_file_name, 'rb') as f_id:
    read_raw_file = np.fromfile(f_id, count=-1, dtype=np.uint8).reshape(-1, 16).T


with open(lab_file_name, 'rb') as f_id:
    read_raw_file = np.fromfile(f_id, count=-1, dtype=np.uint8)

read_raw_file.shape
read_raw_file[:10]
with open(lab_file_name, 'rb') as f_id:
    temp = f_id.read()

temp[2]
len(temp)

read_raw_file.shape

# Size of the fread from matlab is... 16 x 2548269
# And the `readsize` is 40772301.. This is the count (=16 * 2548269)

plt.plot(read_raw_file[0, :526])
n_length = len(read_raw_file)
length_list.append(n_length)

for i_length in length_list:
print(i_length, [i for i in range(0, 256, 1)[1:] if i_length % i == 0])
# print("\t remainder ", i_length - 16 * int(i_length/16))

