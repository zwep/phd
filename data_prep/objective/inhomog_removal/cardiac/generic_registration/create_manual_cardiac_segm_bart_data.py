import nibabel
import os
import numpy as np
import helper.plot_class as hplotc
import helper.array_transf as harray

"""
Elastix was no use somehow...

Doing it differently ourselves...
"""

ddata_rho = '/media/bugger/MyBook/data/simulated/cardiac/bart/sa/rho'
dest_mask = '/media/bugger/MyBook/data/simulated/cardiac/bart/sa/segm_mask'

rho_file_list = os.listdir(ddata_rho)

sel_rho_file = rho_file_list[14]
rho_array = np.load(os.path.join(ddata_rho, sel_rho_file))
mask_obj = hplotc.MaskCreator(rho_array)

simple_mask = mask_obj.mask
dest_file = os.path.join(dest_mask, sel_rho_file)
np.save(dest_file, simple_mask)