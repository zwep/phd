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
Making use of the created overview table... we are going to plot images from the mat files

"""

ddata_mat = '/data/cmr7t3t/cmr7t/RawData_newbatch/data_mat'
ddata_png = '/data/cmr7t3t/cmr7t/RawData_newbatch/png_mat'
ddata_csv = '/data/cmr7t3t/cmr7t/overview_new_batch.csv'

csv_obj = pd.read_csv(ddata_csv)

for ii, i_row in csv_obj.iterrows():
    if i_row['not in previous']:
        i_file = i_row['mat files']
        i_file = i_file + '.mat'
        base_name = hmisc.get_base_name(i_file)
        mat_file = os.path.join(ddata_mat, i_file)
        mat_obj = scipy.io.loadmat(mat_file)
        data_obj = np.squeeze(mat_obj['data'])
        nx, ny, ncard, nloc = data_obj.shape
        print(f" Data shape {data_obj.shape}")
        for iloc in range(nloc):
            loc_name = f"{base_name}_{str(iloc).zfill(2)}"
            plot_file_name = os.path.join(ddata_png, loc_name) + '.png'
            first_card = np.abs(data_obj[:, :, 0, iloc])
            last_card = np.abs(data_obj[:, :, 15, iloc])
            print(f" Data shape {first_card.shape} - {last_card.shape}")
            plot_obj = hplotc.ListPlot([[first_card, last_card]])
            plot_obj.figure.savefig(plot_file_name)
        hplotc.close_all()
