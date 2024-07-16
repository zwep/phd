import getpass
import os
import sys

# Deciding which OS is being used
if getpass.getuser() == 'bugger':
    local_system = True
    manual_mode = True
else:
    import matplotlib as mpl
    mpl.use('Agg')  # Hopefully this makes sure that we can plot/save stuff
    local_system = False
    manual_mode = False

if local_system:
    project_path = "/home/bugger/PycharmProjects/pytorch_in_mri"
    model_path = os.path.join(project_path, 'config_template')
else:
    project_path = "/home/seb/code/pytorch_in_mri"
    model_path = "/home/seb/code/pytorch_in_mri/config_template"


print('Adding to path: ', project_path)
sys.path.append(project_path)

import helper.misc as hmisc
import numpy as np

# ddata = '/data/seb/semireal/.../input'
ddata = '/data/seb/flavio_npy/train/input'
ddata = '/data/seb/semireal/prostate_simulation_rxtx/train/input'
list_files = [os.path.join(ddata, x) for x in os.listdir(ddata)]

all_coil_pos = []
for i_file in list_files:
    A = np.load(i_file)[0]
    # print('Shape of loaded file ', A.shape)
    coilpos = hmisc.get_coil_position(A)
    all_coil_pos.append(coilpos)

print(np.array(all_coil_pos).mean(axis=0))

"""
Result from this is....
[[ 82.11470037 239.8338015 ]
 [ 48.88810861 169.99656679]
 [ 47.68789014  93.52621723]
 [ 71.42993134  20.5744382 ]
 [207.14325843  16.53792135]
 [225.85549313  86.63623596]
 [226.76186017 161.39138577]
 [216.22487516 235.51513733]]


"""

average_coil_pos = np.array(
    [[ 82.11470037, 239.8338015 ],
 [ 48.88810861, 169.99656679],
 [ 47.68789014,  93.52621723],
 [ 71.42993134,  20.5744382 ],
 [207.14325843,  16.53792135],
 [225.85549313,  86.63623596],
 [226.76186017, 161.39138577],
 [216.22487516, 235.51513733]]
)

import matplotlib.pyplot as plt
plt.imshow(np.zeros((256, 256)))
plt.scatter(average_coil_pos[:, 1], average_coil_pos[:, 0], c='k')