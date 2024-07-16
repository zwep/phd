from objective_configuration.reconstruction import DDATA
import os
import helper.misc as hmisc
import helper.plot_class as hplotc
import numpy as np
import torchio.transforms
from objective_helper.reconstruction import convert_direct2cpx

"""

Test some augmentations that we want to apply to the data..

I still need to test what the effect of RandomAffine in image space is on the kspace
"""


torchio_obj = torchio.transforms.RandomAffine(scales=[0.25, 4], translation=[-50, 50, -50, 50, 0, 0], degrees=[-10, 10, -10, 10, 0, 0],
                                isotropic=True, default_pad_value=0, center='origin')


ddata = os.path.join(DDATA, 'mixed', 'train', 'input')
list_files = os.listdir(ddata)
sel_file = os.path.join(ddata, list_files[0])

loaded_array = hmisc.load_array(sel_file, data_key='kspace', sel_slice='mid')

cpx_array = convert_direct2cpx(loaded_array)

import torch
cur_seed = torch.seed()
torch.manual_seed(cur_seed)
torch.manual_seed(cur_seed)

transformed_A = []
for _ in range(25):
    A_transf = torchio_obj(A_tens[None])
    transformed_A.append(A_transf[0, :, :, 0].numpy())

hplotc.SlidingPlot(np.array(transformed_A))
