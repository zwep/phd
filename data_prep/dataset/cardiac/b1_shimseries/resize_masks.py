import numpy as np
import skimage.transform as sktranf
import os

ddata = '/home/bugger/Documents/data/7T/shimseries/masks'
dtarget = '/home/bugger/Documents/data/7T/shimseries/masks_128'

for i_file in os.listdir(ddata):
    input_file = os.path.join(ddata, i_file)
    target_file = os.path.join(dtarget, i_file)

    A = np.load(input_file)
    A_trans = sktranf.resize(A, (128, 128))

    np.save(target_file, A_trans)