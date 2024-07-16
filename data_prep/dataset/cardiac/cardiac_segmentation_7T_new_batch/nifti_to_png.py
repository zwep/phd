import nibabel
import helper.plot_class as hplotc
import helper.misc as hmisc
import scipy.io
import helper.plot_class
import numpy as np
import os
import json
import pandas as pd

"""
Here we make a selection on which files we are going to use

Here we make use of the create JSON that filtered out the duplicated betweeen the old- and new-batch data
"""

# ddata_nifti = '/data/cmr7t3t/cmr7t/RawData_newbatch/data_nifti'
ddata_nifti = '/data/cmr7t3t/cmr7t/RawData_newbatch/data_nifti_ED_ES_crop'
ddata_png = '/data/cmr7t3t/cmr7t/RawData_newbatch/png_nifti'

for i_file in os.listdir(ddata_nifti):
    base_name = hmisc.get_base_name(i_file)
    data_obj = hmisc.load_array(os.path.join(ddata_nifti,i_file))
    nx, ny, ncard = data_obj.shape
    print(f" Data shape {data_obj.shape}")
    loc_name = f"{base_name}"
    plot_file_name = os.path.join(ddata_png, loc_name) + '.png'
    nchan = data_obj.shape[-1]
    if nchan > 1:
        first_card = np.abs(data_obj[:, :, 0])
        last_card = np.abs(data_obj[:, :, 15])
        print(f" Data shape {first_card.shape} - {last_card.shape}")
        plot_obj = hplotc.ListPlot([[first_card, last_card]])
    else:
        print(f" Data shape {data_obj.shape}")
        plot_obj = hplotc.ListPlot(data_obj[:, :, 0])
    plot_obj.figure.savefig(plot_file_name)

