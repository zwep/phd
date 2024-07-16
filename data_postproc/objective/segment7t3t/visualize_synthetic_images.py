import sys
sys.path.append('/home/bugger/PycharmProjects/pytorch_in_mri')
import argparse
import json
import helper.array_transf as harray
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm
import pathlib
import helper.misc as hmisc
import matplotlib.font_manager
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import ImageColor, Image

"""

"""
ddata = '/data/cmr7t3t/mms1_synthesis_220908'
model_list = os.listdir(ddata)


import helper.plot_class as hplotc
for i_model_name_dir in sorted(model_list):
    print("Directory ", i_model_name_dir)
    model_dir_path = os.path.join(ddata, i_model_name_dir)
    epoch_dir = os.listdir(model_dir_path)
    epoch_dir = [x for x in epoch_dir if x.startswith('test_')]
    for i_epoch in epoch_dir:
        img_dir = os.path.join(ddata, i_model_name_dir, i_epoch, 'niftis/cmr3t2cmr7t')
        png_dir = os.path.join(ddata, i_model_name_dir, i_epoch, 'images')
        file_list_nifti = os.listdir(img_dir)
        n_files = len(file_list_nifti)
        multp_six = n_files // 6
        n_files_multp_six = multp_six * 6
        counter = -1
        for index_range in np.split(np.arange(n_files_multp_six), n_files_multp_six//6):
            plot_array = []
            file_string = ''
            for ii in index_range:
                sel_file = file_list_nifti[ii]
                base_name = hmisc.get_base_name(sel_file)
                file_png = os.path.join(img_dir, sel_file)
                img_array = np.squeeze(hmisc.load_array(file_png))
                n_slice = img_array.shape[2]
                sel_array = img_array[:, :, n_slice//2]
                plot_array.append(sel_array)
                file_string += base_name + "_"
            dest_file_name = os.path.join(png_dir, 'collage_' + file_string[:-1] + '.png')
            plot_array = np.stack(plot_array)
            fig_obj = hplotc.ListPlot([plot_array], cmap='gray', debug=True,
                                      sub_col_row=(2, 3), ax_off=True, wspace=0, hspace=0,
                                      figsize=(10, 15), aspect='auto')
            fig_obj.figure.savefig(dest_file_name, bbox_inches='tight', pad_inches=0.0)
            hplotc.close_all()
